"""
src/project_registry.py

Auto-builds a project registry at process startup by scanning knowledge_base/
and extracting the GitHub URL from the first H1 heading of each file.

SCALABILITY CONTRACT
--------------------
To add a new project to the registry:
  1. Drop a new .md file into knowledge_base/.
  2. Make sure its first H1 line contains the project URL anywhere in the text.

Example first line formats that all work:
    # My Project — source code: https://github.com/user/repo
    # My Project and it's source code link is https://github.com/user/repo
    # My Project (https://github.com/user/repo)

No code changes needed.  The registry is rebuilt on every process start.

DEDUPLICATION
-------------
Multiple KB files that share the same URL (e.g. the five Retail Lakehouse
files) are treated as one project in the UI button list.  In citations, the
LLM receives the project name + URL with each chunk and is instructed to
deduplicate.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.config import KB_PATH

logger = logging.getLogger(__name__)

# Matches any http/https URL, stopping at whitespace or common trailing punctuation.
_URL_RE = re.compile(r"https?://[^\s\)\],;]+")

# Patterns to strip from the H1 text when producing a clean display name.
# Applied in order — the first match wins.
_NOISE_RES: list[re.Pattern] = [
    re.compile(r"\s+and\s+it[''`]?s?\s+source\s+code\s+link\s+is\s+https?://\S+", re.IGNORECASE),
    re.compile(r"\s*[—\-–]+\s*source\s*:?\s*https?://\S+", re.IGNORECASE),
    re.compile(r"\s*source\s*:?\s*https?://\S+", re.IGNORECASE),
    re.compile(r"\s*\(?https?://\S+\)?"),   # fallback: bare URL in parens or plain
    re.compile(r"\s*https?://\S+"),          # last resort: any remaining URL
]


def _extract_from_h1(file_path: Path) -> tuple[str, str] | None:
    """Read the first H1 heading and return ``(display_name, url)`` or ``None``."""
    try:
        with file_path.open(encoding="utf-8") as fh:
            frontmatter_url = ""
            for raw_line in fh:
                line = raw_line.strip()
                if line.startswith("source_url:"):
                    frontmatter_url = line.partition(":")[2].strip()
                if not line.startswith("#"):
                    continue
                # Found the first heading — look for a URL.
                url_match = _URL_RE.search(line)
                if url_match is None:
                    if frontmatter_url:
                        name = re.sub(r"^#+\s*", "", line).strip()
                        return (name or file_path.stem, frontmatter_url)
                    logger.warning("Project file %s has no GitHub URL in H1 — skipping", file_path.name)
                    return None
                url = url_match.group(0).rstrip(".,;)")

                # Build display name: strip markup prefix then noise patterns.
                name = re.sub(r"^#+\s*", "", line).strip()
                for noise in _NOISE_RES:
                    name, n_subs = noise.subn("", name)
                    if n_subs:
                        break
                name = name.strip(" -—–")

                return (name or file_path.stem, url)
    except OSError as exc:
        logger.warning("Cannot read %s: %s", file_path, exc)
    return None


def _build() -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    """Scan KB_PATH and build the registry + deduplicated project list.

    Two-pass approach:
    1. Collect every file mapping into ``registry`` and group by URL.
    2. For each unique URL, pick the best display name by preferring files
       whose stem ends with ``_main`` (the top-level README for a project).
       Falls back to the first alphabetically-sorted entry if no ``_main``
       file exists.

    Returns
    -------
    registry : dict[str, dict]
        Maps each file stem to ``{display_name, url}``.
    projects : list[dict]
        Deduplicated ``{display_name, url}`` entries, sorted by display name.
        Used to render the "View on GitHub" buttons in the UI.
    """
    registry: dict[str, dict[str, str]] = {}
    # url → list of (stem, display_name) for all files sharing that URL
    url_entries: dict[str, list[tuple[str, str]]] = {}

    for md_file in sorted((KB_PATH / "projects").rglob("*.md")):
        parsed = _extract_from_h1(md_file)
        if parsed is None:
            logger.warning(
                "Skipping project %s from registry — no GitHub URL in H1", md_file.name
            )
            continue
        display_name, url = parsed
        registry[md_file.stem] = {"display_name": display_name, "url": url}
        url_entries.setdefault(url, []).append((md_file.stem, display_name))

    # Pick the best display name per URL: prefer _main files, else first alphabetical.
    projects: list[dict[str, str]] = []
    for url, entries in url_entries.items():
        main_entries = [e for e in entries if e[0].endswith("_main")]
        stem, display_name = main_entries[0] if main_entries else entries[0]
        projects.append({"display_name": display_name, "url": url})

    projects.sort(key=lambda p: p["display_name"])
    logger.info(
        "Project registry: %d file mappings → %d unique projects",
        len(registry),
        len(projects),
    )
    return registry, projects


# ── Module-load-time initialisation ─────────────────────────────────────────
_REGISTRY, _PROJECTS = _build()


def lookup(source: str) -> dict[str, str]:
    """Return ``{display_name, url}`` for a KB file stem.

    Falls back to ``{display_name: source, url: ""}`` so callers never crash
    on an unregistered file.
    """
    return _REGISTRY.get(source, {"display_name": source, "url": ""})


def get_all_projects() -> list[dict[str, str]]:
    """Return deduplicated ``{display_name, url}`` for all known projects.

    Sorted alphabetically by display name.  Used to render GitHub buttons.
    """
    return list(_PROJECTS)
