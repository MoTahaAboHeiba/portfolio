"""
scripts/ingest.py

One-shot ingestion script — run locally to populate Qdrant Cloud.
This script is NEVER run on Hugging Face Spaces at startup.

Usage (from the project root):
    python scripts/ingest.py

Prerequisites:
    - .env file with QDRANT_URL and QDRANT_API_KEY (GROQ/GEMINI not needed here)
    - pip install -r requirements.txt
"""

import logging
import sys
import time
import uuid
from pathlib import Path

from google import genai

# Add the project root to sys.path so that `from src.config import ...` resolves
# correctly regardless of the current working directory when the script is run.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.chunking import split_text  # noqa: E402  (import after sys.path patch)
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

from src.config import settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
COLLECTION_NAME = "motaha-ai"
VECTOR_SIZE = 3072
MODEL_NAME = "gemini-embedding-001"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
SEPARATORS = ["\n\n", "\n", ". ", " "]
KB_PATH = _PROJECT_ROOT / "knowledge_base"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _deterministic_uuid(file_path: str, chunk_index: int) -> str:
    """Generate a stable UUID from the file path and chunk index.

    Using uuid5 ensures that re-running ingest on the same content produces the
    same point IDs, making the upsert idempotent.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_path}:{chunk_index}"))


def _ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection if it does not already exist."""
    existing_names = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing_names:
        info = client.get_collection(COLLECTION_NAME)
        current_dim = getattr(info.config.params.vectors, "size", None)
        if current_dim != VECTOR_SIZE:
            logger.warning(
                "Collection '%s' exists with dim=%s but Gemini requires dim=%s; deleting and recreating it.",
                COLLECTION_NAME,
                current_dim,
                VECTOR_SIZE,
            )
            client.delete_collection(collection_name=COLLECTION_NAME)
        else:
            logger.info("Collection '%s' already exists with the correct dim=%d — upserting into it", COLLECTION_NAME, VECTOR_SIZE)
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info("Created collection '%s' (dim=%d, metric=Cosine)", COLLECTION_NAME, VECTOR_SIZE)


def _embed_document(text: str, client: genai.Client) -> list[float]:
    result = client.models.embed_content(
        model=MODEL_NAME,
        contents=text,
        config={"task_type": "RETRIEVAL_DOCUMENT"},
    )
    return result.embeddings[0].values


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    gemini_client = genai.Client(api_key=settings.gemini_api_key)
    logger.info("Connecting to Qdrant Cloud at %s", settings.QDRANT_URL)
    client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60)

    _ensure_collection(client)

    # split_text is the pure-Python equivalent of RecursiveCharacterTextSplitter
    logger.info("Loading Gemini embedding model: %s", MODEL_NAME)

    # ── Load and chunk all markdown files ────────────────────────────────────
    all_chunks: list[dict] = []
    files_processed = 0

    for md_file in sorted(KB_PATH.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        file_path = str(md_file.relative_to(KB_PATH))
        source = md_file.stem

        file_chunks = split_text(content, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=SEPARATORS)
        for idx, chunk_text in enumerate(file_chunks):
            all_chunks.append(
                {
                    "id": _deterministic_uuid(file_path, idx),
                    "text": chunk_text,
                    "source": source,
                    "file_path": file_path,
                    "chunk_index": idx,
                }
            )
        files_processed += 1
        logger.info("  %-40s → %d chunks", md_file.name, len(file_chunks))

    logger.info("Total chunks to embed: %d", len(all_chunks))

    # ── Embed ─────────────────────────────────────────────────────────────────
    texts = [c["text"] for c in all_chunks]

    BATCH_SIZE = 90  # safe margin below the 100/min limit
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        logger.info(
            f"Embedding batch {i // BATCH_SIZE + 1} of "
            f"{-(-len(texts) // BATCH_SIZE)} "
            f"({len(batch)} chunks)"
        )
        embeddings.extend([_embed_document(text, gemini_client) for text in batch])
        if i + BATCH_SIZE < len(texts):
            logger.info("Rate limit pause: sleeping 65 seconds...")
            time.sleep(65)

    logger.info("Embedding complete (%d vectors, dim=%d)", len(embeddings), VECTOR_SIZE)

    # ── Build Qdrant points and upsert ────────────────────────────────────────
    points = [
        PointStruct(
            id=chunk["id"],
            vector=embedding,
            payload={
                "text": chunk["text"],
                "source": chunk["source"],
                "file_path": chunk["file_path"],
                "chunk_index": chunk["chunk_index"],
            },
        )
        for chunk, embedding in zip(all_chunks, embeddings)
    ]

    BATCH_SIZE = 50
    upserted = 0
    for batch_start in range(0, len(points), BATCH_SIZE):
        batch = points[batch_start : batch_start + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        upserted += len(batch)
        logger.info("  Upserted %d / %d points", upserted, len(points))


    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 52)
    print("  Ingestion complete")
    print(f"  Files processed : {files_processed}")
    print(f"  Total chunks    : {len(all_chunks)}")
    print(f"  Upserted count  : {len(points)}")
    print(f"  Collection      : {COLLECTION_NAME}")
    print(f"  Vector size     : {VECTOR_SIZE}")
    print("=" * 52)
    print()


if __name__ == "__main__":
    main()
