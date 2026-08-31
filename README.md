---
title: MoTaha AI
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.26.0
python_version: '3.11'
app_file: app.py
pinned: false
---

# MoTaha AI

A RAG-powered portfolio chatbot that answers grounded questions about
Mohamed Taha's engineering background, projects, skills, and architecture
decisions.

Ask a question — get a cited, context-grounded answer.

## Architecture

| Layer | Technology |
|---|---|
| UI | Gradio `ChatInterface` (streaming) |
| Dense retrieval | Qdrant Cloud + FastEmbed `BAAI/bge-small-en-v1.5` (ONNX) |
| Sparse retrieval | BM25Okapi (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion (k = 60) |
| Primary LLM | Groq — `openai/gpt-oss-120b` (streaming) |
| Fallback LLM | Gemini — `gemini-1.5-flash` |
| Config | `pydantic-settings` |

## Project Structure

```
├── app.py                    # Gradio entrypoint
├── requirements.txt
├── .env.example
├── knowledge_base/           # Markdown knowledge files (do not modify)
├── scripts/
│   └── ingest.py             # Run locally once to populate Qdrant Cloud
└── src/
    ├── config.py             # pydantic-settings singleton + KB_PATH
    ├── pipeline.py           # End-to-end RAG orchestration
    ├── retrieval/
    │   ├── dense.py          # Qdrant Cloud vector search
    │   ├── sparse.py         # BM25 index (built at import time)
    │   ├── fusion.py         # RRF
    │   └── retriever.py      # Parallel dense + sparse → RRF → top-k
    └── generation/
        ├── prompt.py         # System prompt template
        ├── scope_guard.py    # Refuse when evidence score < 0.35
        └── generator.py      # Groq streaming → Gemini fallback
```

## Local Development

### Prerequisites

- Python 3.11+
- A [Qdrant Cloud](https://cloud.qdrant.io/) cluster (free tier works)
- Groq API key — [console.groq.com](https://console.groq.com)
- Gemini API key — [aistudio.google.com](https://aistudio.google.com)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd motaha-ai

# Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copy the environment template and fill in your keys
cp .env.example .env

# Ingest the knowledge base into Qdrant Cloud (run once)
python scripts/ingest.py

# Start the Gradio app
python app.py
```

## Deployment on Hugging Face Spaces

1. Create a new Space with the **Gradio** SDK.
2. Push this repository to the Space (or sync via GitHub Actions).
3. Set the following **Secrets** in the Space settings:
   - `GROQ_API_KEY`
   - `GEMINI_API_KEY`
   - `QDRANT_API_KEY`
   - `QDRANT_URL`
4. Add a **Variable** (not a secret): `GRADIO_SSR_MODE=false`

   Gradio 6 turns on server-side rendering (SSR) on Spaces by default. The Node
   sidecar often exits immediately after boot and kills the app — logs show
   `with SSR ⚡` followed by `Stopping Node.js server...`. Disabling SSR fixes
   this. `app.py` also sets the env var at import time; the Space variable is a
   backup because Hugging Face may override launch kwargs.

ZeroGPU is supported even though this app does not run GPU inference: the
retrieval stack uses FastEmbed's ONNX backend. The app intentionally does not
run model code on the GPU, but the chat event is marked with
`@spaces.GPU(duration=120)` because ZeroGPU Spaces require at least one
decorated function during startup. The decorator lets the event run through
the ZeroGPU queue while retrieval remains CPU/ONNX-based. `app.py` also disables Gradio's
`share` tunnel on Spaces, since Spaces already supplies the public URL and the
tunnel can terminate the process after startup.

The `app.py` file is detected automatically by the Gradio SDK. No Dockerfile
or server configuration is required.

## License

MIT
