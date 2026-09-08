# MoTaha AI

> An evidence-grounded portfolio chatbot for Mohamed Taha's data and AI
> engineering work.

MoTaha AI turns a static engineering portfolio into a conversational interface.
Visitors can ask about projects, architecture decisions, technologies,
certifications, skills, and career experience. Answers are generated from the
repository's Markdown knowledge base and include links back to the relevant
portfolio projects.

The application is designed to be honest about its evidence. If retrieval does
not find sufficiently relevant context, the pipeline refuses to guess instead
of presenting an unsupported answer.

## What it does

- Answers questions about the documented portfolio knowledge base.
- Supports follow-up questions using recent conversation history.
- Handles greetings, introductions, and small talk without invoking retrieval.
- Uses hybrid retrieval: Gemini semantic vectors plus BM25 keyword search.
- Combines retrieval rankings with Reciprocal Rank Fusion (RRF).
- Applies a semantic scope guard before calling an LLM.
- Streams responses over Server-Sent Events (SSE).
- Falls back from Groq to Gemini when the primary generation provider fails.
- Detects Arabic user messages and responds in Egyptian Arabic (Ammiya).
- Applies RTL layout to Arabic user and assistant bubbles in the frontend.
- Renders project/source chips that link to GitHub and portfolio sections.

## Architecture

```text
Browser
  │
  │ POST /chat
  ▼
FastAPI + Uvicorn
  │
  ├── Query classifier
  │     └── canned response for greetings, introductions, and small talk
  │
  └── RAG pipeline
        ├── Conversation-aware query augmentation
        ├── Dense retrieval
        │     ├── Gemini gemini-embedding-001
        │     └── Qdrant Cloud collection: motaha-ai
        ├── Sparse retrieval
        │     └── BM25Okapi over the local knowledge base
        ├── Reciprocal Rank Fusion
        ├── Scope guard
        │     └── requires top dense cosine score >= 0.35
        ├── Generation
        │     ├── Groq: openai/gpt-oss-120b, streaming
        │     └── Gemini: gemini-1.5-flash, fallback
        └── Programmatic project/source metadata
```

### Technology stack

| Area | Technology |
| --- | --- |
| Backend API | FastAPI |
| Server | Uvicorn |
| Frontend | Plain HTML, CSS, and browser JavaScript |
| Markdown rendering | Marked.js via CDN |
| Semantic embeddings | Gemini `gemini-embedding-001` |
| Vector database | Qdrant Cloud |
| Keyword retrieval | `rank-bm25` |
| Rank fusion | Reciprocal Rank Fusion, `k=60` |
| Primary LLM | Groq `openai/gpt-oss-120b` |
| Fallback LLM | Gemini `gemini-1.5-flash` |
| Configuration | `pydantic-settings` |
| Tests | Pytest |

## Repository layout

```text
.
├── index.html                    # Browser chat UI and streaming client
├── main.py                       # FastAPI application and SSE endpoint
├── requirements.txt              # Runtime and test dependencies
├── .env.example                  # Environment variable template
├── Procfile                      # Generic process start command
├── render.yaml                   # Render web-service configuration
├── knowledge_base/
│   ├── identity/                 # About, career focus, value proposition
│   ├── professional/             # Skills, education, certifications
│   └── projects/                 # Project-specific source documents
├── scripts/
│   └── ingest.py                 # Embeds Markdown and upserts Qdrant points
├── src/
│   ├── classifier.py              # Query classification and canned replies
│   ├── chunking.py                # Shared recursive Markdown chunking
│   ├── config.py                  # Environment-backed settings
│   ├── pipeline.py                # End-to-end RAG orchestration
│   ├── project_registry.py        # Project names and GitHub URL discovery
│   ├── session.py                 # Conversation history utilities
│   ├── generation/
│   │   ├── generator.py            # Groq generation and Gemini fallback
│   │   ├── prompt.py               # Grounding, formatting, and language rules
│   │   └── scope_guard.py          # Evidence threshold and refusal response
│   └── retrieval/
│       ├── dense.py                # Gemini embeddings and Qdrant search
│       ├── sparse.py               # Process-local BM25 index
│       ├── fusion.py               # RRF implementation
│       └── retriever.py             # Parallel dense/sparse orchestration
└── tests/                         # Unit and integration tests
```

## Requirements

- Python 3.12+ recommended. Python 3.11+ is expected to work.
- A Qdrant Cloud collection populated by the ingestion script.
- A Groq API key for the primary provider.
- A Gemini API key for embeddings and fallback generation.
- A Qdrant API key and cluster URL.

## Local setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in the values:

```dotenv
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
QDRANT_API_KEY=your_qdrant_key
QDRANT_URL=https://your-cluster.qdrant.io
```

Do not commit `.env`, API keys, or credential backups. Environment variables
are loaded by `src/config.py`.

### 4. Build or refresh the Qdrant index

Run this after adding or changing knowledge-base Markdown files:

```bash
python scripts/ingest.py
```

The ingestion process:

1. Reads every `.md` file under `knowledge_base/`.
2. Splits documents into 512-character chunks with 64-character overlap.
3. Embeds documents with Gemini `gemini-embedding-001`.
4. Creates or validates the `motaha-ai` Qdrant collection.
5. Upserts deterministic point IDs into Qdrant.

The collection uses 3,072-dimensional cosine vectors. Re-ingestion can recreate
an existing collection if its vector dimension does not match the active
embedding model.

### 5. Start the application

Development server with auto-reload:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

The equivalent module entrypoint is:

```bash
python main.py
```

## HTTP API

### `GET /`

Returns the frontend from `index.html`.

### `GET /health`

Returns a lightweight process health response:

```json
{"status": "ok"}
```

This endpoint confirms that the web process is running. It does not validate
Qdrant reachability or LLM provider credentials.

### `POST /chat`

Request:

```json
{
  "message": "What projects have you built?",
  "history": [
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am a Data Engineer..."}
  ]
}
```

The response is an SSE stream with `Content-Type: text/event-stream`. Text
chunks use the format:

```text
data: response text

```

The stream ends with:

```text
data: [DONE]

```

An empty `message` returns HTTP 400.

## Response behavior and grounding

The pipeline follows this order:

1. Classify the query.
2. Return a canned response for non-career conversation.
3. Augment follow-up questions with recent history.
4. Run dense and sparse retrieval concurrently.
5. Fuse candidates with RRF and keep the top five chunks.
6. Reject low-confidence retrieval when the best dense score is below `0.35`.
7. Generate a grounded answer with Groq.
8. Fall back to Gemini if Groq fails.
9. Append source metadata programmatically.

The LLM prompt requires first-person answers, prohibits unsupported claims, and
keeps source formatting outside the model's responsibility.

## Arabic and English support

Language is inferred from the user's message; there is no language selector.

- Arabic input receives Egyptian Arabic (Ammiya).
- English input receives English.
- Technical names such as `FastAPI`, `Qdrant`, `RAG`, and `Delta Lake` stay in
  English.
- Arabic bubbles use Cairo font, RTL direction, right alignment, and increased
  line height.
- Source chips remain in English because they represent filenames and links.

## Deployment

### Render

The repository includes `render.yaml` and a `Procfile`. A Render web service
can use:

```text
Build command: pip install -r requirements.txt
Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Configure these environment variables in the service dashboard:

- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `QDRANT_API_KEY`
- `QDRANT_URL`

### Other ASGI hosts

Any host that can run an ASGI application can use:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

The application serves its own frontend, so a separate static-site deployment
is not required.

## Testing and verification

Run the currently available test suite with:

```bash
python -m pytest -q
```

For a quick application smoke check:

```bash
python -c "from fastapi.testclient import TestClient; from main import app; print(TestClient(app).get('/health').json())"
```

Before releasing a deployment, verify:

- `/health` returns `{"status": "ok"}`.
- The frontend loads from `/`.
- An in-scope question streams a response and sources.
- An out-of-scope question receives a grounded refusal.
- Arabic responses are RTL and English responses remain LTR.
- Qdrant collection dimensions match `gemini-embedding-001`.

## Current review notes

This repository contains legacy tests and CI references for an earlier
`motaha_ai` package architecture. The active runtime uses `main.py` and the
`src/` package shown above. The GitHub workflow also invokes `uv sync`, but the
repository currently has no `pyproject.toml` or `uv.lock`; that workflow needs
to be migrated to the active `requirements.txt` setup before it can be treated
as a reliable CI gate.

## Security notes

- Never commit `.env`, API tokens, private keys, or credential backups.
- Rotate any provider credential that has ever been exposed outside the secret
  manager.
- Replace the permissive `allow_origins=["*"]` CORS setting in `main.py` with
  the exact production frontend origins before deploying a public service.
- `/chat` currently accepts client-provided history. Treat it as conversation
  context, not as trusted evidence; factual answers are still constrained by
  retrieved knowledge-base context.
- Add authentication, rate limiting, request-size limits, and structured
  observability before using the service beyond a portfolio/demo workload.

## License

MIT

## Author

**Mohamed Taha Abo Heiba** — Data Engineer, Cairo, Egypt  
[Portfolio](https://motahaaboheiba.github.io) |
[LinkedIn](https://linkedin.com/in/mohamed-taha-abo-heiba) |
[GitHub](https://github.com/MoTahaAboHeiba)
