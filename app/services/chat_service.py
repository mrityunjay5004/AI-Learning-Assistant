"""
Business logic for POST /chat - the RAG chat endpoint.

Pipeline:
  1. Verify the roadmap exists.
  2. Retrieve the top relevant roadmap chunks for the user's message from the
     vector store (semantic retrieval, not full-roadmap dumping - this is
     what makes it a real RAG system rather than in-memory context
     injection), with similarity filtering and dedup applied by the
     retriever (see app/rag/retriever.py).
  3. Pull recent conversation history from the DB (product-thinking feature:
     multi-turn memory so follow-up questions like "what about the second
     one?" resolve correctly).
  4. Build a grounded, citation-labeled prompt and call the LLM for a
     structured JSON answer.
  5. Persist both the user message and assistant reply for future turns.
"""
import logging
import time

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.exceptions import DatabaseError, RoadmapNotFoundError
from app.models.db_models import ChatMessage, Roadmap
from app.models.schemas import ChatRequest, ChatResponse, LLMChatOutput
from app.prompts import CHAT_SYSTEM_INSTRUCTION, build_chat_prompt
from app.rag.retriever import format_context, retrieve
from app.services.llm_client import generate_structured

logger = logging.getLogger(__name__)
settings = get_settings()

MAX_HISTORY_TURNS = 6  # most recent user+assistant turns included as context


def _format_history(messages: list[ChatMessage]) -> str:
    if not messages:
        return ""
    lines = [f"{m.role.capitalize()}: {m.content}" for m in messages]
    return "\n".join(lines)


def handle_chat(db: Session, req: ChatRequest) -> ChatResponse:
    roadmap = db.get(Roadmap, req.roadmap_id)
    if roadmap is None:
        raise RoadmapNotFoundError(f"No roadmap found with id '{req.roadmap_id}'.")

    # 1. Retrieve relevant chunks (semantic search, scoped to this roadmap;
    # low-similarity hits and duplicate task chunks are already filtered out
    # by the retriever).
    retrieved_chunks = retrieve(roadmap.id, req.message, top_k=settings.rag_top_k)
    if not retrieved_chunks:
        logger.warning(
            "No relevant context retrieved for roadmap_id=%s message=%r; "
            "falling back to general-knowledge answer.",
            roadmap.id,
            req.message,
        )
    roadmap_context = format_context(retrieved_chunks)

    # 2. Pull recent conversation history for multi-turn continuity
    try:
        recent_messages = (
            db.query(ChatMessage)
            .filter(ChatMessage.roadmap_id == roadmap.id)
            .order_by(ChatMessage.created_at.desc())
            .limit(MAX_HISTORY_TURNS)
            .all()
        )
    except SQLAlchemyError as exc:
        logger.exception("Failed to load chat history for roadmap_id=%s", roadmap.id)
        raise DatabaseError("Failed to load conversation history.") from exc

    recent_messages.reverse()  # chronological order
    history_str = _format_history(recent_messages)

    # 3. Build grounded prompt and call the LLM
    prompt = build_chat_prompt(
        roadmap_context=roadmap_context,
        conversation_history=history_str,
        user_message=req.message,
    )

    llm_start = time.monotonic()
    llm_output: LLMChatOutput = generate_structured(
        system_instruction=CHAT_SYSTEM_INSTRUCTION,
        user_prompt=prompt,
        output_schema=LLMChatOutput,
    )
    llm_latency_ms = (time.monotonic() - llm_start) * 1000
    logger.info(
        "Chat LLM call for roadmap_id=%s completed in %.0fms (retrieved_chunks=%d)",
        roadmap.id,
        llm_latency_ms,
        len(retrieved_chunks),
    )

    # 4. Persist the turn (user message + assistant reply) for future context
    try:
        db.add(ChatMessage(roadmap_id=roadmap.id, role="user", content=req.message))
        db.add(ChatMessage(roadmap_id=roadmap.id, role="assistant", content=llm_output.response))
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to persist chat turn for roadmap_id=%s", roadmap.id)
        raise DatabaseError("Failed to save the conversation turn.") from exc

    return ChatResponse(
        response=llm_output.response,
        follow_up_questions=llm_output.follow_up_questions,
        retrieved_context=[f"[{c.source_label}] {c.text}" for c in retrieved_chunks],
    )
