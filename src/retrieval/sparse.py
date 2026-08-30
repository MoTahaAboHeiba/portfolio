"""
src/retrieval/sparse.py

Sparse BM25 retrieval over the knowledge base corpus.

The BM25 index is built once at module load time (when this module is first
imported) so it is ready for every subsequent query with zero latency.
Chunking parameters are kept identical to scripts/ingest.py to ensure the
BM25 token space matches the stored Qdrant chunks.
"""

import logging
from typing import Any

from src.chunking import split_text
from rank_bm25 import BM25Okapi

from src.config import KB_PATH

logger = logging.getLogger(__name__)

# ── Chunking parameters (must match scripts/ingest.py exactly) ───────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SEPARATORS = ["\n\n", "\n", ". ", " "]


def _build_index() -> tuple[list[str], list[dict[str, Any]], BM25Okapi]:
    """Load every .md file from knowledge_base/, chunk it, and build the index."""
    # split_text is the pure-Python equivalent of RecursiveCharacterTextSplitter

    chunks: list[str] = []
    metadata_list: list[dict[str, Any]] = []

    for md_file in sorted(KB_PATH.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        file_path = str(md_file.relative_to(KB_PATH))
        source = md_file.stem

        file_chunks = split_text(content, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=SEPARATORS)
        for idx, chunk_text in enumerate(file_chunks):
            chunks.append(chunk_text)
            metadata_list.append(
                {
                    "source": source,
                    "file_path": file_path,
                    "chunk_index": idx,
                }
            )
    if not chunks:
        raise RuntimeError(
            f"BM25 index is empty. No .md files found under {KB_PATH}. "
            "Ensure knowledge_base/ is committed and pushed to the repository."
        )
    tokenized = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)

    logger.info(
        "BM25 index built: %d chunks from %d files in %s",
        len(chunks),
        len(list(KB_PATH.rglob("*.md"))),
        KB_PATH,
    )
    return chunks, metadata_list, bm25


# Module-load-time initialisation — runs once per process.
_chunks, _metadata, _bm25 = _build_index()


def query(text: str, top_k: int = 10) -> list[tuple[str, float, dict[str, Any]]]:
    """Return the *top_k* chunks most relevant to *text* by BM25 score.

    Returns
    -------
    list[tuple[str, float, dict]]
        Each element is ``(chunk_text, bm25_score, metadata)``.
    """
    tokenized_query = text.lower().split()
    scores = _bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    return [
        (_chunks[idx], float(scores[idx]), _metadata[idx])
        for idx in top_indices
    ]
