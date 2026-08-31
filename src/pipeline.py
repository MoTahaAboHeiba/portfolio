"""
src/pipeline.py

End-to-end RAG pipeline for MoTaha AI.

Query classification runs first. Greetings, introductions, and small
talk receive canned human responses and never touch the retrieval stack.

Genuine career questions go through:
  1. Hybrid retrieval (dense + BM25 + RRF)
  2. Scope guard (cosine threshold 0.35)
  3. LLM generation (Groq primary, Gemini fallback)
  4. Sources block (appended in Python, not by the LLM)
"""

# Reindex required if the Qdrant collection was built with a different embedding
# dimension than the active Gemini embedding model. gemini-embedding-001 emits
# 3072-d vectors, so the collection must be recreated or rebuilt to match.

from __future__ import annotations

import logging
from typing import Generator

from src.classifier import classify, get_canned_response, CAREER_CATEGORY
from src.generation.generator import generate
from src.generation.scope_guard import REFUSAL, is_sufficient
from src.project_registry import lookup
from src.retrieval.retriever import retrieve
from src.session import ConversationTurn, SessionService

logger = logging.getLogger(__name__)


def _build_sources_block(
    results: list[dict],
    score_threshold: float = 0.005,
) -> str:
    """Build the 📎 Sources line from retrieval results.

    Deduplicates by source stem. Returns empty string if no sources.
    """
    seen: set[str] = set()
    parts: list[str] = []

    for result in results:
        if result.get("score", 0) < score_threshold:
            continue
        stem = result.get("source", "")
        if not stem or stem in seen:
            continue
        seen.add(stem)
        entry = lookup(stem)
        display_name = entry.get("display_name") or stem
        url = entry.get("url", "")
        if url:
            parts.append(f"[{display_name}]({url})")
        else:
            parts.append(display_name)

    if not parts:
        return ""
    return "You can find it here: " + " · ".join(parts)


def answer(
    query: str,
    history: list | None = None,
) -> Generator[str, None, None]:
    """Yield response tokens for a user query.

    Never yields sources on refusals or non-career queries.
    """
    turns = SessionService.from_gradio_history(history or [])

    # ── Step 1: classify ──────────────────────────────────────────────────────
    category = classify(query, history=history)
    if category != CAREER_CATEGORY:
        logger.info("Query classified as %s — returning canned response", category)
        yield get_canned_response(category, query)
        return

    # ── Step 2: retrieve ──────────────────────────────────────────────────────
    augmented_query = SessionService.build_augmented_query(query, turns)
    try:
        retrieval_result = retrieve(augmented_query)
        results = retrieval_result["chunks"]
        top_dense_score = retrieval_result["top_dense_score"]
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc, exc_info=True)
        yield (
            "I ran into a technical issue retrieving context. "
            "Please try again in a moment."
        )
        return

    # ── Step 3: scope guard ───────────────────────────────────────────────────
    if not is_sufficient(top_dense_score):
        logger.info(
            "Scope guard fired — top_dense_score=%.4f below threshold",
            top_dense_score,
        )
        yield REFUSAL
        return

    # ── Step 4: generate ──────────────────────────────────────────────────────
    try:
        for chunk in generate(query, results, history=history):
            yield chunk
    except Exception as exc:
        logger.error("Generation failed: %s", exc, exc_info=True)
        yield (
            "I ran into a technical issue generating a response. "
            "Please try again in a moment."
        )
        return

    # ── Step 5: sources (only on real answers) ────────────────────────────────
    sources_block = _build_sources_block(results, score_threshold=0.005)
    if sources_block:
        yield "\n\n" + sources_block
