"""
High-level RAG retrieval interface used by the chat service.

index_roadmap(): chunk -> embed -> upsert (called once, right after a
roadmap is generated).

retrieve(): embed the user's query (with task_type="retrieval_query", which
Gemini's embedding model optimizes differently than document embeddings),
fetch nearest-neighbor chunks scoped to that roadmap_id, then apply three
retrieval-quality passes before handing results to the prompt builder:

  1. Similarity filtering - Chroma's cosine *distance* (0 = identical,
     larger = less similar) is thresholded, so a roadmap with few chunks
     doesn't force in weakly-related context just to fill top_k slots.
  2. Deduplication - guards against the same task chunk appearing twice
     (e.g. if overlapping chunk types are added later) by task_index.
  3. Result cap - even after filtering, hard-cap at MAX_RESULTS so a very
     permissive threshold can't flood the prompt.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.rag import vector_store
from app.rag.chunking import chunk_roadmap
from app.services.llm_client import embed_text

logger = logging.getLogger(__name__)

# Cosine distance threshold: hits farther than this are considered
# irrelevant and discarded rather than padding the prompt with noise.
MAX_DISTANCE = 0.8
# Hard cap on chunks returned, independent of the caller's requested top_k.
MAX_RESULTS = 5


@dataclass
class RetrievedChunk:
    text: str
    source_label: str
    distance: float


def index_roadmap(
    roadmap_id: str,
    goal_title: str,
    estimated_hours: int,
    skills: list[str],
    tasks: list[dict],
) -> None:
    start = time.monotonic()
    chunks = chunk_roadmap(roadmap_id, goal_title, estimated_hours, skills, tasks)
    embeddings = [embed_text(c["text"], task_type="retrieval_document") for c in chunks]
    vector_store.upsert_chunks(chunks, embeddings)
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "Indexed roadmap_id=%s into vector store (%d chunks, %.0fms)",
        roadmap_id,
        len(chunks),
        elapsed_ms,
    )


def retrieve(roadmap_id: str, user_message: str, top_k: int) -> list[RetrievedChunk]:
    start = time.monotonic()
    query_embedding = embed_text(user_message, task_type="retrieval_query")
    raw_hits = vector_store.query(roadmap_id, query_embedding, top_k=max(top_k, MAX_RESULTS))

    seen_task_indices: set[int] = set()
    results: list[RetrievedChunk] = []

    for hit in raw_hits:
        if hit["distance"] > MAX_DISTANCE:
            continue

        task_index = hit["metadata"].get("task_index", -1)
        if task_index in seen_task_indices:
            continue
        seen_task_indices.add(task_index)

        results.append(
            RetrievedChunk(
                text=hit["text"],
                source_label=hit["metadata"].get("source_label", "Roadmap"),
                distance=hit["distance"],
            )
        )
        if len(results) >= MAX_RESULTS:
            break

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "Retrieval for roadmap_id=%s: %d/%d hits kept after filtering (%.0fms)",
        roadmap_id,
        len(results),
        len(raw_hits),
        elapsed_ms,
    )
    return results


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a labeled block for the chat prompt, so
    the model can cite a specific task by name (e.g. "as covered in Task
    2/5") instead of speaking about the roadmap generically."""
    if not chunks:
        return "(no relevant roadmap context was retrieved for this question)"
    return "\n".join(f"[{c.source_label}] {c.text}" for c in chunks)
