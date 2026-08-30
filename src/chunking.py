"""
src/chunking.py

Pure-Python recursive character text splitter.

Mirrors the behaviour of LangChain's RecursiveCharacterTextSplitter without
any external dependencies.  Used by both scripts/ingest.py (at ingestion time)
and src/retrieval/sparse.py (at BM25 index build time) so the chunk boundaries
are identical in both places.

Algorithm
---------
1. Split the text on the first separator.
2. For each resulting piece:
   - If it fits within chunk_size  → accumulate it.
   - If it is too large            → recurse with the remaining separators.
3. Merge accumulated pieces into final chunks of at most chunk_size characters,
   carrying a chunk_overlap-character tail from the previous chunk into the next.
"""

from __future__ import annotations


def _join_len(items: list[str], sep: str) -> int:
    """Exact character length of ``sep.join(items)``."""
    if not items:
        return 0
    return sum(len(s) for s in items) + len(sep) * (len(items) - 1)


def _merge_splits(
    splits: list[str],
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Merge a flat list of small *splits* into overlapping chunks.

    Parameters
    ----------
    splits:
        Pre-validated pieces, each individually ≤ chunk_size.
    separator:
        String used to re-join pieces when measuring length.
    chunk_size:
        Maximum character count per output chunk.
    chunk_overlap:
        Number of characters from the end of one chunk to carry into the next.
    """
    chunks: list[str] = []
    current: list[str] = []

    for s in splits:
        if not s:
            continue
        if _join_len(current + [s], separator) > chunk_size and current:
            # Emit the accumulated chunk.
            text = separator.join(current)
            if text.strip():
                chunks.append(text)
            # Drop pieces from the front until the tail fits within chunk_overlap.
            while current and _join_len(current, separator) > chunk_overlap:
                current.pop(0)
        current.append(s)

    if current:
        text = separator.join(current)
        if text.strip():
            chunks.append(text)

    return chunks


def split_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: list[str] | None = None,
) -> list[str]:
    """Split *text* into chunks of at most *chunk_size* characters.

    Parameters
    ----------
    text:
        Input text to split.
    chunk_size:
        Maximum character count per chunk.
    chunk_overlap:
        Characters of overlap between consecutive chunks.
    separators:
        Ordered list of separators to try, from coarsest to finest.
        Defaults to ``["\\n\\n", "\\n", ". ", " ", ""]``.

    Returns
    -------
    list[str]
        Non-empty chunks, each at most *chunk_size* characters (or slightly
        over only when a single token exceeds chunk_size with no separator).
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    sep = separators[0]
    next_seps = separators[1:] if len(separators) > 1 else [""]

    raw_pieces = text.split(sep) if sep else list(text)

    good: list[str] = []   # pieces that already fit
    result: list[str] = []

    for piece in raw_pieces:
        if not piece:
            continue
        if len(piece) <= chunk_size:
            good.append(piece)
        else:
            # Flush accumulated good pieces first.
            if good:
                result.extend(_merge_splits(good, sep, chunk_size, chunk_overlap))
                good = []
            # Recurse on the oversized piece with a finer separator.
            result.extend(split_text(piece, chunk_size, chunk_overlap, next_seps))

    if good:
        result.extend(_merge_splits(good, sep, chunk_size, chunk_overlap))

    return result
