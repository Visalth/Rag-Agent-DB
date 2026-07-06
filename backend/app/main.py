import json
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import GroqError
from pinecone.exceptions import PineconeException
from pydantic import BaseModel

from .config import CORS_ORIGINS
from .parsing import extract_blocks, chunk_blocks
from . import rag, llm

app = FastAPI(title="RAG Docs Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_NS = "default"

UPSTREAM_ERRORS = (requests.RequestException, PineconeException, GroqError)
UNAVAILABLE = "A service this app depends on is temporarily unavailable. Try again in a moment."


class ChatRequest(BaseModel):
    question: str
    session: str = DEFAULT_NS
    source: str | None = None  # limit answers to one uploaded file


def _sources(hits: list[dict]) -> list[str]:
    by_src: dict[str, set] = {}
    for h in hits:
        if not h["source"]:
            continue
        pages = by_src.setdefault(h["source"], set())
        if h.get("page") is not None:
            pages.add(h["page"])
    out = []
    for src in sorted(by_src):
        pages = sorted(by_src[src])
        out.append(f"{src} (p. {', '.join(map(str, pages))})" if pages else src)
    return out


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...), session: str = Form(DEFAULT_NS)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        blocks = extract_blocks(file.filename, data)
    except ValueError as e:
        raise HTTPException(415, str(e))

    chunks = chunk_blocks(blocks)
    if not chunks:
        raise HTTPException(422, "No readable text found in file")

    try:
        stored = rag.store_chunks(chunks, source=file.filename, namespace=session)
    except UPSTREAM_ERRORS:
        raise HTTPException(502, UNAVAILABLE)
    return {"file": file.filename, "chunks": stored}


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Empty question")
    try:
        hits = rag.retrieve(question, namespace=req.session, source=req.source)
        reply = llm.answer(question, hits)
    except UPSTREAM_ERRORS:
        raise HTTPException(502, UNAVAILABLE)
    return {"answer": reply, "sources": _sources(hits)}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Empty question")
    try:
        hits = rag.retrieve(question, namespace=req.session, source=req.source)
    except UPSTREAM_ERRORS:
        raise HTTPException(502, UNAVAILABLE)

    def events():
        yield f"data: {json.dumps({'type': 'sources', 'sources': _sources(hits)})}\n\n"
        try:
            for piece in llm.answer_stream(question, hits):
                yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n"
        except UPSTREAM_ERRORS:
            yield f"data: {json.dumps({'type': 'error', 'message': UNAVAILABLE})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.delete("/reset")
def reset(session: str = DEFAULT_NS):
    try:
        rag.clear_namespace(session)
    except UPSTREAM_ERRORS:
        raise HTTPException(502, UNAVAILABLE)
    return {"cleared": session}
