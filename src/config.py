"""
src/config.py

Central configuration loaded from environment variables via pydantic-settings.
All four keys are required. If any is missing at import time, pydantic raises a
clear ValidationError before any query can be served.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GROQ_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_URL: str | None = None

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False,
    }

    @property
    def groq_api_key(self) -> str | None:
        return self.GROQ_API_KEY

    @property
    def gemini_api_key(self) -> str | None:
        return self.GEMINI_API_KEY

    @property
    def qdrant_api_key(self) -> str | None:
        return self.QDRANT_API_KEY

    @property
    def qdrant_url(self) -> str | None:
        return self.QDRANT_URL

    @property
    def missing_required(self) -> list[str]:
        required = [
            "GROQ_API_KEY",
            "GEMINI_API_KEY",
            "QDRANT_API_KEY",
            "QDRANT_URL",
        ]
        return [name for name in required if not getattr(self, name, None)]

    def ensure_runtime_ready(self) -> None:
        missing = self.missing_required
        if missing:
            raise RuntimeError(
                "Missing required environment variables: "
                + ", ".join(missing)
                + ". Add them in your Space secrets/settings and redeploy."
            )


# Singleton — imported by every module that needs config or the KB path.
settings = Settings()

# Canonical path to the knowledge base directory.
# Path(__file__) is src/config.py → .parent is src/ → .parent is project root.
KB_PATH: Path = Path(__file__).parent.parent / "knowledge_base"
