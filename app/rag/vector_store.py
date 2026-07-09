"""
Persistent vector storage using Chroma.

Design decision: Chroma is chosen over a plain in-memory dict + cosine
similarity loop because:
  - It persists to disk (CHROMA_PERSIST_DIR), so roadmaps survive app
    restarts without re-embedding.
  - It gives us a real ANN index and metadata filtering for free, which
    matters if the number of roadmaps/chunks grows.
  - It's embeddable (no separate server process to run), keeping the
    assignment's setup to "pip install and run".

We embed chunks ourselves via Gemini (see services/llm_client.embed_text)
rather than using Chroma's default embedding function, so embedding model
choice is explicit and consistent between indexing and querying.

Each roadmap gets its own logical namespace via metadata filtering (rather
than one Chroma collection per roadmap) to avoid unbounded collection
creation, while still allowing `where={"roadmap_id": ...}` to scope
retrieval strictly to that roadmap's knowledge base.
"""
from __future__ import annotations

import logging

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

settings = get_settings()

_client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)
_collection = _client.get_or_create_collection(
    name="roadmap_chunks",
    metadata={"hnsw:space": "cosine"},
)


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    if not chunks:
        return
    try:
        _collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in chunks],
        )
    except Exception as exc:
        logger.error("Vector store upsert failed: %s", exc)
        raise VectorStoreError("Failed to index roadmap content for retrieval.") from exc
    logger.info("Upserted %d chunks into vector store", len(chunks))


def query(roadmap_id: str, query_embedding: list[float], top_k: int) -> list[dict]:
    """Query the vector store and return each hit's document text, metadata,
    and cosine distance, so the caller (retriever) can filter by relevance,
    deduplicate, and attach a citeable source label - rather than the store
    silently making those decisions."""
    try:
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"roadmap_id": roadmap_id},
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("Vector store query failed for roadmap_id=%s: %s", roadmap_id, exc)
        raise VectorStoreError("Failed to retrieve roadmap context.") from exc

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def has_roadmap(roadmap_id: str) -> bool:
    results = _collection.get(where={"roadmap_id": roadmap_id}, limit=1)
    return bool(results.get("ids"))
