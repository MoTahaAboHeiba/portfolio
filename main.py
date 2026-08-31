from __future__ import annotations

import inspect

from google import genai
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.pipeline import answer


client = genai.Client(api_key=settings.gemini_api_key)

app = FastAPI(title="MoTaha AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your Render service URL (e.g. https://motaha-ai.onrender.com)
    # and your Cloudflare Pages URL before deploying to production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
async def root():
    return FileResponse("index.html")


def _stream_sync(generator):
    for chunk in generator:
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_async(generator):
    async for chunk in generator:
        yield f"data: {chunk}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(request: Request):
    payload = await request.json()
    message = str(payload.get("message", "")).strip() if payload else ""
    history = payload.get("history", []) if payload else []

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    result = answer(message, history=history)

    if inspect.isasyncgen(result):
        return StreamingResponse(
            _stream_async(result),
            media_type="text/event-stream",
        )

    if inspect.isgenerator(result):
        return StreamingResponse(
            _stream_sync(result),
            media_type="text/event-stream",
        )

    return StreamingResponse(
        _stream_sync(iter([str(result)])),
        media_type="text/event-stream",
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
