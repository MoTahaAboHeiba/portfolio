"""
src/classifier.py

Classifies incoming queries before they reach the RAG pipeline.
Handles greetings, personal introductions, and small talk with canned
human responses so the pipeline only processes genuine career questions.

Categories
----------
GREETING      : hi, hello, hey, good morning, salaam, marhaba, etc.
INTRODUCTION  : "I'm Mostafa", "my name is X", "I'm a recruiter from Y"
SMALL_TALK    : how are you, thanks, nice to meet you, etc.
CAREER        : anything else — goes to the full RAG pipeline
"""

from __future__ import annotations

import re

# ── Constants ─────────────────────────────────────────────────────────────────

CAREER_CATEGORY = "CAREER"

# ── Patterns ──────────────────────────────────────────────────────────────────

_GREETING = re.compile(
    r"^(hi|hey|hello|good\s+(morning|afternoon|evening|day)|howdy|"
    r"greetings|salaam|marhaba|ahlan|سلام|مرحبا|أهلاً|هاي|هلو|صباح|مساء)"
    r"[\s!,،.]*$",
    re.IGNORECASE,
)

_GREETING_WITH_NAME = re.compile(
    r"^(hi|hey|hello|good\s+(morning|afternoon|evening|day)|salaam|مرحبا|هاي)"
    r"[\s,،!]*\w+[\s!,،.]*$",
    re.IGNORECASE,
)

_INTRODUCTION = re.compile(
    r"(i[''`]?m\s+[a-zA-Z\u0600-\u06FF]+|"
    r"my name is\s+[a-zA-Z\u0600-\u06FF]+|"
    r"i am\s+a?\s*(recruiter|hiring manager|engineer|developer|student|"
    r"hr|talent|founder|cto|tech lead)|"
    r"nice to meet you|"
    r"pleased to meet|"
    r"introducing myself)",
    re.IGNORECASE,
)

_SMALL_TALK = re.compile(
    r"^(how are you|how('?re| are) (you|things|it going)|"
    r"what('?s| is) up|wassup|sup|"
    r"thank(s| you)|thx|ty|"
    r"great|awesome|cool|nice|got it|"
    r"ok|okay|alright|sounds good|"
    r"see you|bye|goodbye|take care)[\s!.،]*$",
    re.IGNORECASE,
)


# ── Canned responses ──────────────────────────────────────────────────────────

_GREETING_RESPONSE = (
    "Hey! I'm Mohamed Taha — a Data Engineer and AI Engineer based in Egypt.\n\n"
    "Feel free to ask me about my projects, skills, certifications, or anything "
    "about my engineering background. What would you like to know?"
)

_SMALL_TALK_RESPONSE = (
    "Doing well, appreciate you asking!\n\n"
    "Is there something specific about my work or engineering background "
    "I can help you with?"
)

_THANKS_RESPONSE = (
    "Happy to help.\n\n"
    "If there is anything else you want to know about my projects or "
    "background, just ask."
)

_GOODBYE_RESPONSE = (
    "Good talking to you. If you want to reach out directly:\n"
    "Email: mohamed-aboheiba@outlook.com\n"
    "LinkedIn: linkedin.com/in/mohamed-taha-abo-heiba"
)

_THANKS_RE = re.compile(
    r"^(thank(s| you)|thx|ty)[\s!.،]*$", re.IGNORECASE
)
_GOODBYE_RE = re.compile(
    r"^(bye|goodbye|see you|take care|cya)[\s!.،]*$", re.IGNORECASE
)

_FOLLOWUP = re.compile(
    r"^(tell me more|what about (that|this)|"
    r"can you (explain|elaborate|expand)|"
    r"go on|continue|and\?|more details?|"
    r"what do you mean|clarify that)[\s?.,]*$",
    re.IGNORECASE,
)

_FOLLOW_UP_START = re.compile(
    r"^(and|what about|how about|why|how did|how do|how|tell me more|"
    r"can you elaborate|also|more on|go on|continue|elaborate|explain more)\b",
    re.IGNORECASE,
)

_PRONOUN_FOLLOW_UP = re.compile(
    r"^(it|that|this|they|them|those|there)\??$",
    re.IGNORECASE,
)


def _extract_name(text: str) -> str | None:
    """Try to extract a first name from an introduction message."""
    patterns = [
        re.compile(r"i[''`]?m\s+([a-zA-Z\u0600-\u06FF]+)", re.IGNORECASE),
        re.compile(r"my name is\s+([a-zA-Z\u0600-\u06FF]+)", re.IGNORECASE),
        re.compile(
            r"(?:hi|hey|hello)[,\s]+i[''`]?m\s+([a-zA-Z\u0600-\u06FF]+)",
            re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).capitalize()
    return None


def _is_follow_up(text: str) -> bool:
    """Detect short follow-up questions that need prior conversation context."""
    stripped = text.strip()
    words = stripped.split()

    if _FOLLOW_UP_START.search(stripped) or _PRONOUN_FOLLOW_UP.match(stripped):
        return True
    if len(words) <= 8 and "?" in stripped and len(words) <= 5:
        return True
    return len(words) <= 3


def classify(text: str, history: list | None = None) -> str:
    """Return the query category: GREETING, INTRODUCTION, SMALL_TALK, or CAREER."""
    stripped = text.strip()

    if history and _FOLLOWUP.match(stripped):
        return CAREER_CATEGORY

    if (
        history
        and _is_follow_up(stripped)
        and not _SMALL_TALK.match(stripped)
        and not _THANKS_RE.match(stripped)
        and not _GOODBYE_RE.match(stripped)
    ):
        return CAREER_CATEGORY

    if _GREETING.match(stripped) or _GREETING_WITH_NAME.match(stripped):
        # A greeting with a name might also be an introduction — check both.
        if _INTRODUCTION.search(stripped):
            return "INTRODUCTION"
        return "GREETING"

    if _INTRODUCTION.search(stripped):
        return "INTRODUCTION"

    if _SMALL_TALK.match(stripped):
        return "SMALL_TALK"

    return CAREER_CATEGORY


def get_canned_response(category: str, text: str) -> str:
    """Return the appropriate canned response for a non-career query."""
    if category == "GREETING":
        return _GREETING_RESPONSE

    if category == "INTRODUCTION":
        name = _extract_name(text)
        if name:
            return (
                f"Hey {name}! Nice to connect.\n\n"
                "I'm Mohamed Taha — a Data Engineer and AI Engineer. "
                "Ask me anything about my work, projects, or technical "
                "background. I'm happy to answer."
            )
        return (
            "Nice to meet you!\n\n"
            "I'm Mohamed Taha — a Data Engineer and AI Engineer. "
            "Feel free to ask about my projects, skills, or engineering "
            "background."
        )

    if category == "SMALL_TALK":
        if _THANKS_RE.match(text.strip()):
            return _THANKS_RESPONSE
        if _GOODBYE_RE.match(text.strip()):
            return _GOODBYE_RESPONSE
        return _SMALL_TALK_RESPONSE

    # Should not reach here — CAREER is handled by the pipeline.
    return _GREETING_RESPONSE
