"""
Thin wrapper around the Gemini API that centralizes:

- Prompt/response plumbing (system instruction + JSON mime type)
- Retry with exponential backoff on transient failures (tenacity)
- Robust JSON extraction (models sometimes wrap JSON in ```json fences, add
  stray commentary, leave a trailing comma before a closing bracket, or -
  rarely - truncate output mid-object) and Pydantic validation of the result
- A repair pass: if the first response fails schema validation, we send the
  invalid output back to the model with the validation error and ask it to
  fix it, before giving up and raising LLMGenerationError.
- Per-call latency logging, and a distinct LLMTimeoutError so slow-model
  failures are visibly different from malformed-output failures.

This is the one place in the codebase that talks to google-generativeai, so
swapping providers (OpenAI/Claude) later only means rewriting this file.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Type, TypeVar

import google.generativeai as genai
from google.api_core.exceptions import DeadlineExceeded
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.exceptions import LLMGenerationError, LLMTimeoutError, EmbeddingError

logger = logging.getLogger(__name__)

settings = get_settings()
genai.configure(api_key=settings.google_api_key)

T = TypeVar("T", bound=BaseModel)

# Trailing comma before a closing ] or } - a common near-miss from models
# that isn't valid JSON but is trivially fixable without another round trip.
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")


class _TransientLLMError(Exception):
    """Raised internally to trigger a tenacity retry (network/5xx errors)."""


def _strip_markdown_fence(text: str) -> str:
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return text


def _remove_trailing_commas(text: str) -> str:
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def _extract_balanced_object(text: str) -> str | None:
    """Scan for the first '{' and return the substring up to its matching
    '}', tracking string/escape state so braces inside string values don't
    throw off the balance count. Handles cases where the model wraps valid
    JSON in explanatory prose before/after it."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None  # unbalanced - likely truncated output


def _extract_json(raw_text: str) -> dict:
    """Extract a JSON object from a raw LLM string, tolerating the common
    failure modes: markdown code fences, leading/trailing prose, trailing
    commas, and a JSON object embedded inside other text. Truncated/partial
    JSON (unbalanced braces) is NOT silently guessed at - it raises, so the
    self-repair loop in generate_structured() asks the model to resend a
    complete object rather than us fabricating missing fields.
    """
    text = _strip_markdown_fence(raw_text.strip())

    # Fast path: the whole string is already valid JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Next: same string with trailing commas removed.
    cleaned = _remove_trailing_commas(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Next: extract a brace-balanced object from within surrounding prose,
    # then retry both as-is and with trailing commas stripped.
    candidate = _extract_balanced_object(text)
    if candidate is not None:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            candidate_cleaned = _remove_trailing_commas(candidate)
            return json.loads(candidate_cleaned)  # let this raise if still invalid

    raise json.JSONDecodeError(
        "No complete JSON object found in LLM output (possibly truncated)", text, 0
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_TransientLLMError),
)
def _call_model(model: genai.GenerativeModel, prompt: str) -> str:
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
            request_options={"timeout": 30},
        )

        print("\n" + "=" * 80)
        print("RAW GEMINI RESPONSE")
        print("=" * 80)

        try:
            print(response.text)
        except Exception:
            print(response)

        print("=" * 80 + "\n")

    except DeadlineExceeded as exc:
        logger.error("Gemini call timed out: %s", exc)
        raise LLMTimeoutError() from exc

    except Exception as exc:
        import traceback

        print("\n" + "=" * 80)
        print("GEMINI EXCEPTION")
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80 + "\n")

        logger.warning("Transient error calling Gemini: %s", exc)
        raise _TransientLLMError(str(exc)) from exc

    if not response.candidates:
        raise _TransientLLMError("Empty response (no candidates) from Gemini")

    return response.text


def generate_structured(
    system_instruction: str,
    user_prompt: str,
    output_schema: Type[T],
    max_repair_attempts: int = 2,
) -> T:
    """Call Gemini and return a validated instance of `output_schema`.

    Flow:
      1. Ask the model for JSON matching the schema.
      2. Parse (tolerating markdown fences / trailing commas / surrounding
         prose) + validate. On success, return.
      3. On parse/validation failure, send the error back to the model and
         ask it to correct its own output (bounded number of repair loops).
      4. If still failing, raise LLMGenerationError. Timeouts propagate
         immediately as LLMTimeoutError rather than being retried as
         schema-repair attempts.
    """
    model = genai.GenerativeModel(
    model_name=settings.gemini_model,
    system_instruction=system_instruction,
)

    prompt = user_prompt
    last_error: Exception | None = None
    call_start = time.monotonic()

    for attempt in range(max_repair_attempts + 1):
        try:
            raw_text = _call_model(model, prompt)
            parsed = _extract_json(raw_text)
            validated = output_schema.model_validate(parsed)
            logger.info(
                "LLM structured generation succeeded for %s in %.0fms (attempt %d/%d)",
                output_schema.__name__,
                (time.monotonic() - call_start) * 1000,
                attempt + 1,
                max_repair_attempts + 1,
            )
            return validated
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "LLM output for %s failed validation (attempt %s/%s): %s",
                output_schema.__name__,
                attempt + 1,
                max_repair_attempts + 1,
                exc,
            )
            prompt = (
                f"{user_prompt}\n\n"
                "Your previous response was invalid JSON or did not match the "
                f"required schema. Error: {exc}\n"
                "Return ONLY a single complete, valid JSON object matching the "
                "schema. No markdown fences, no commentary, no trailing commas."
            )
        except LLMTimeoutError:
            raise  # don't burn repair attempts on a timeout - surface it directly
        except _TransientLLMError as exc:
            last_error = exc
            logger.error("Gemini call exhausted retries: %s", exc)
            break

    total_ms = (time.monotonic() - call_start) * 1000
    logger.error(
        "Giving up on LLM structured generation for %s after %.0fms: %s",
        output_schema.__name__,
        total_ms,
        last_error,
    )
    raise LLMGenerationError(
        "The AI model returned an invalid response after multiple attempts."
    )


def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    try:
        result = genai.embed_content(
            model=settings.gemini_embedding_model,
            content=text,
        )

        print("\n================ EMBEDDING RESULT ================\n")
        print(result)
        print("\n==================================================\n")

        return result["embedding"]

    except Exception as exc:
        import traceback

        print("\n================ EMBEDDING ERROR ================\n")
        traceback.print_exc()
        print("\n=================================================\n")

        logger.error("Embedding generation failed: %s", exc)
        raise EmbeddingError() from exc