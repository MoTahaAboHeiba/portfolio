"""
src/retrieval/fusion.py

Reciprocal Rank Fusion (RRF) over dense and sparse result lists.

RRF score formula:
    score(d) = Σ  1 / (k + rank(d, list))
               lists

where rank is 1-indexed and k=60 is the RRF smoothing constant.

Deduplication is performed by chunk text before scoring so a chunk that
appears in both lists is merged into a single entry with a combined score.
"""

from typing import Any


def rrf(
    dense_results: list[tuple[str, float, dict[str, Any]]],
    sparse_results: list[tuple[str, float, dict[str, Any]]],
    k: int = 60,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Fuse *dense_results* and *sparse_results* using Reciprocal Rank Fusion.

    Parameters
    ----------
    dense_results:
        Ordered list of ``(text, score, metadata)`` from the dense retriever.
    sparse_results:
        Ordered list of ``(text, score, metadata)`` from the sparse retriever.
    k:
        RRF smoothing constant.  Default is 60 (standard value from the
        original RRF paper by Cormack et al., 2009).

    Returns
    -------
    list[tuple[str, float, dict]]
        Deduplicated, merged list sorted by RRF score descending.
        Score is the sum of ``1 / (k + rank)`` contributions across lists.
        Rank is 1-indexed (top result has rank 1).
    """
    rrf_scores: dict[str, float] = {}
    chunk_store: dict[str, tuple[str, dict[str, Any]]] = {}

    for rank, (text, _score, meta) in enumerate(dense_results, start=1):
        rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (k + rank)
        chunk_store.setdefault(text, (text, meta))

    for rank, (text, _score, meta) in enumerate(sparse_results, start=1):
        rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (k + rank)
        chunk_store.setdefault(text, (text, meta))

    sorted_texts = sorted(rrf_scores, key=lambda t: rrf_scores[t], reverse=True)

    return [
        (chunk_store[t][0], rrf_scores[t], chunk_store[t][1])
        for t in sorted_texts
    ]
