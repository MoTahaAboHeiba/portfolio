"""
src/retrieval/retriever.py

Hybrid retrieval orchestrator: runs dense and sparse retrieval in parallel,
then fuses results with RRF and returns the top-k final chunks.

Return value also carries ``top_dense_score`` — the highest cosine similarity
score from the Qdrant results before any fusion.  This is used by the scope
guard to decide whether there is meaningful semantic overlap between the query
and the knowledge base, independently of the RRF fusion artifact.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.retrieval import dense, sparse, fusion
from src.project_registry import lookup

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    dense_top_k: int = 10,
    sparse_top_k: int = 10,
    final_top_k: int = 5,
) -> dict[str, Any]:
    """Run hybrid retrieval and return fused results plus the top dense score.

    Dense (Qdrant Cloud) and sparse (BM25) retrieval are executed in parallel
    using a thread pool.  Results are fused with Reciprocal Rank Fusion and
    trimmed to *final_top_k*.

    Parameters
    ----------
    query:
        Raw user question string.
    dense_top_k:
        Number of candidates to fetch from the dense vector index.
    sparse_top_k:
        Number of candidates to fetch from the BM25 sparse index.
    final_top_k:
        Maximum number of fused results to return.

    Returns
    -------
    dict with two keys:
        ``chunks`` : list[dict[str, Any]]
            Top-*final_top_k* fused chunks.  Each dict contains:
                - ``text``      : chunk content
                - ``score``     : RRF score (float, range ≈ 0–0.033)
                - ``source``    : filename stem (e.g. "EduMate-RAG")
                - ``file_path`` : relative path within knowledge_base/
        ``top_dense_score`` : float
            The highest cosine similarity score returned by the Qdrant search
            before fusion.  Range 0–1.  Used by the scope guard.
            0.0 if the dense search returned no results.
    """
    dense_results: list[tuple[str, float, dict]] = []
    sparse_results: list[tuple[str, float, dict]] = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_dense = executor.submit(dense.search, query, dense_top_k)
        future_sparse = executor.submit(sparse.query, query, sparse_top_k)

        for future in as_completed([future_dense, future_sparse]):
            if future is future_dense:
                dense_results = future.result()
                logger.debug("Dense retrieval returned %d results", len(dense_results))
            else:
                sparse_results = future.result()
                logger.debug("Sparse retrieval returned %d results", len(sparse_results))

    # Extract the top cosine score BEFORE fusion — this is the value the
    # scope guard checks against the 0.35 threshold.
    top_dense_score: float = dense_results[0][1] if dense_results else 0.0

    fused = fusion.rrf(dense_results, sparse_results, k=60)
    top = fused[:final_top_k]

    logger.info(
        "Retrieval complete — dense: %d (top cosine: %.3f), sparse: %d, fused top-%d: %d",
        len(dense_results),
        top_dense_score,
        len(sparse_results),
        final_top_k,
        len(top),
    )

    return {
        "chunks": [
            {
                "text": text,
                "score": score,
                "source": meta.get("source", ""),
                "file_path": meta.get("file_path", ""),
                "project_name": lookup(meta.get("source", ""))["display_name"],
                "project_url": lookup(meta.get("source", ""))["url"],
            }
            for text, score, meta in top
        ],
        "top_dense_score": top_dense_score,
    }
