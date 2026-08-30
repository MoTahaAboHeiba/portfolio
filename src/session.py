"""
src/session.py

Conversation memory utilities for MoTaha AI.

Provides turn normalization, query augmentation for retrieval, and
history formatting for LLM prompts. Used by the Gradio app (Phase 1)
and available for future session-based API endpoints.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class ConversationTurn:
    user_question: str
    assistant_answer: str


@dataclass
class ConversationSession:
    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)


class SessionService:
    """In-memory session store and conversation formatting helpers."""

    def __init__(self, max_turns: int = 10, max_history_tokens: int = 800) -> None:
        self.max_turns = max_turns
        self.max_history_tokens = max_history_tokens
        self._sessions: dict[str, ConversationSession] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = ConversationSession(session_id=session_id)
        return session_id

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def get_session(self, session_id: str) -> ConversationSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationSession(session_id=session_id)
        return self._sessions[session_id]

    def update_session(
        self, session_id: str, turn: ConversationTurn
    ) -> ConversationSession:
        session = self.get_session(session_id)
        session.turns.append(turn)
        if len(session.turns) > self.max_turns:
            session.turns = session.turns[-self.max_turns :]
        return session

    def delete_session(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False
        del self._sessions[session_id]
        return True

    @staticmethod
    def strip_sources(answer: str) -> str:
        """Remove the sources block appended by the pipeline."""
        marker = "\n\n📎 Sources:"
        idx = answer.find(marker)
        if idx != -1:
            return answer[:idx]
        return answer

    @staticmethod
    def from_gradio_history(history: list) -> list[ConversationTurn]:
        """Convert Gradio ChatInterface history to ConversationTurn list."""
        turns: list[ConversationTurn] = []
        if not history:
            return turns

        for item in history:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                user_msg, bot_msg = item[0], item[1]
                if user_msg:
                    turns.append(
                        ConversationTurn(
                            user_question=str(user_msg),
                            assistant_answer=str(bot_msg or ""),
                        )
                    )
            elif isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                if role == "user" and content:
                    turns.append(
                        ConversationTurn(user_question=str(content), assistant_answer="")
                    )
                elif role == "assistant" and content and turns:
                    turns[-1] = ConversationTurn(
                        user_question=turns[-1].user_question,
                        assistant_answer=str(content),
                    )

        return turns

    @staticmethod
    def build_augmented_query(
        question: str, history: list[ConversationTurn]
    ) -> str:
        """Combine recent turns with a follow-up into a standalone retrieval query."""
        if not history:
            return question

        recent = history[-2:]
        parts: list[str] = []
        for turn in recent:
            answer = SessionService.strip_sources(turn.assistant_answer)
            if len(answer) > 200:
                answer = answer[:200]
            parts.append(f"Q: {turn.user_question}")
            parts.append(f"A: {answer}")
        parts.append(f"Follow-up: {question}")
        return "\n".join(parts)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text.split())

    def format_history_for_prompt(self, history: list[ConversationTurn]) -> str:
        """Format turns for prompt injection, respecting the token budget."""
        if not history:
            return ""

        header = "## Conversation History"
        remaining = self.max_history_tokens - self._estimate_tokens(header)
        selected: list[str] = []

        for turn in reversed(history):
            block = (
                f"User: {turn.user_question}\n"
                f"Assistant: {self.strip_sources(turn.assistant_answer)}"
            )
            block_tokens = self._estimate_tokens(block)
            if block_tokens > remaining and selected:
                break
            selected.insert(0, block)
            remaining -= block_tokens
            if remaining <= 0:
                break

        return header + "\n" + "\n".join(selected)

    def trim_history_for_llm(
        self, history: list[ConversationTurn]
    ) -> list[ConversationTurn]:
        """Return the newest turns that fit within the token budget."""
        if not history:
            return []

        selected: list[ConversationTurn] = []
        remaining = self.max_history_tokens

        for turn in reversed(history):
            block = (
                f"User: {turn.user_question}\n"
                f"Assistant: {self.strip_sources(turn.assistant_answer)}"
            )
            block_tokens = self._estimate_tokens(block)
            if block_tokens > remaining and selected:
                break
            selected.insert(0, turn)
            remaining -= block_tokens
            if remaining <= 0:
                break

        return selected
