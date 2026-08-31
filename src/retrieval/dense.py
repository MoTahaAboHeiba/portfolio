"""
src/retrieval/dense.py

Dense vector retrieval against Qdrant Cloud using Gemini embeddings.

The model and the Qdrant client are each lazily initialised on first call and
then reused for the lifetime of the process.
"""

import logging
from typing import Any

from google import genai
from qdrant_client import QdrantClient

from src.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "motaha-ai"
MODEL_NAME = "gemini-embedding-001"

_client: QdrantClient | None = None
_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client


def embed_query(text: str) -> list[float]:
    client = _get_gemini_client()
    result = client.models.embed_content(
        model=MODEL_NAME,
        contents=text,
        config={"task_type": "RETRIEVAL_QUERY"},
    )
    return result.embeddings[0].values


def embed_document(text: str) -> list[float]:
    client = _get_gemini_client()
    result = client.models.embed_content(
        model=MODEL_NAME,
        contents=text,
        config={"task_type": "RETRIEVAL_DOCUMENT"},
    )
    return result.embeddings[0].values


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        logger.info("Qdrant client connected to %s", settings.QDRANT_URL)
    return _client


def search(
    query: str,
    top_k: int = 10,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Embed *query* and return the top-k closest chunks from Qdrant Cloud.

    Returns
    -------
    list[tuple[str, float, dict]]
        Each element is ``(chunk_text, cosine_score, payload_metadata)``.
    """
    if not settings.QDRANT_URL or not settings.QDRANT_API_KEY:
        raise RuntimeError(
            "Qdrant settings are not configured. Set QDRANT_URL and QDRANT_API_KEY in the Space secrets."
        )

    client = _get_client()
    query_vector = embed_query(query)

    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        hits = response.points
    else:
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

    return [
        (hit.payload.get("text", ""), hit.score, hit.payload)
        for hit in hits
    ]
