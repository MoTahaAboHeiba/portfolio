"""
src/generation/generator.py

LLM generation with Groq as the primary provider and Gemini as fallback.

Primary  : Groq — model openai/gpt-oss-120b, streaming.
Fallback : Gemini — model gemini-1.5-flash, non-streaming (yields single chunk).

The function is a generator so the caller (pipeline.py / app.py) can stream
tokens directly to the Gradio ChatInterface without buffering.
"""

import logging
from typing import Generator, Any

import warnings

import groq

# google.generativeai emits a FutureWarning about migrating to google-genai.
# The package still works correctly; suppress the warning to keep logs clean.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai


from src.config import settings
from src.generation.prompt import SYSTEM_PROMPT
from src.session import ConversationTurn, SessionService

logger = logging.getLogger(__name__)

GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-1.5-flash"


def _format_history_for_llm(history: list | None) -> str:
    if not history:
        return ""
    lines: list[str] = []
    for turn in history[-4:]:
        role = "Visitor" if turn.get("role") == "user" else "You"
        content = str(turn.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def generate(
    query: str,
    context_chunks: list[dict[str, Any]],
    history: list | None = None,
) -> Generator[str, None, None]:
    """Stream an answer grounded in *context_chunks*.

    Parameters
    ----------
    query:
        The raw user question.
    context_chunks:
        Ordered list of retrieval dicts, each containing at minimum a ``text``
        key. Produced by ``retriever.retrieve()``.
    history:
        Prior conversation turns for multi-turn context.

    Yields
    ------
    str
        Incremental text tokens (Groq) or the full response (Gemini fallback).
    """
    if not settings.GROQ_API_KEY and not settings.GEMINI_API_KEY:
        raise RuntimeError(
            "No LLM API keys are configured. Set GROQ_API_KEY and/or GEMINI_API_KEY in the Space secrets."
        )

    def _chunk_header(chunk: dict[str, Any]) -> str:
        name = chunk.get("project_name", chunk.get("source", "Unknown"))
        url = chunk.get("project_url", "")
        return f"[Source: {name} | {url}]" if url else f"[Source: {name}]"

    context = "\n---\n".join(
        f"{_chunk_header(chunk)}\n{chunk['text']}"
        for chunk in context_chunks
    )

    system_content = SYSTEM_PROMPT.format(context=context)
    history_text = _format_history_for_llm(history)

    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    if history_text:
        messages.append({
            "role": "user",
            "content": f"[Previous conversation]\n{history_text}",
        })
        messages.append({
            "role": "assistant",
            "content": "Understood. I have the conversation context.",
        })
    messages.append({"role": "user", "content": query})

    # ── Primary: Groq streaming ──────────────────────────────────────────────
    try:
        client = groq.Groq(api_key=settings.GROQ_API_KEY)
        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            stream=True,
        )
        logger.info("Provider: Groq (%s)", GROQ_MODEL)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        return

    except Exception as exc:
        logger.warning("Groq failed — %s. Falling back to Gemini.", exc)

    # ── Fallback: Gemini non-streaming ───────────────────────────────────────
    logger.info("Gemini fallback used — query length %d", len(query))
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    if history_text:
        full_prompt = f"{system_content}\n\n[Previous conversation]\n{history_text}\n\nUser: {query}"
    else:
        full_prompt = f"{system_content}\n\nUser: {query}"
    try:
        response = model.generate_content(full_prompt)
        yield response.text
    except AttributeError:
        response = model.generate_content(full_prompt)
        logger.warning("Gemini response.text unavailable; falling back to content.parts[0].text")
        yield response.candidates[0].content.parts[0].text
    except Exception as exc:
        logger.exception("Gemini fallback failed: %s", exc)
        yield "I ran into a technical issue generating a response. Please try again in a moment."
