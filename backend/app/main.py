import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import CORS_ORIGINS
from .parsing import extract_text, chunk_text
from . import rag, llm

app = FastAPI(title="RAG Docs Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_NS = "default"


class ChatRequest(BaseModel):
    question: str
    session: str = DEFAULT_NS


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...), session: str = Form(DEFAULT_NS)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        text = extract_text(file.filename, data)
    except ValueError as e:
        raise HTTPException(415, str(e))

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(422, "No readable text found in file")

    stored = rag.store_chunks(chunks, source=file.filename, namespace=session)
    return {"file": file.filename, "chunks": stored}


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Empty question")
    hits = rag.retrieve(question, namespace=req.session)
    reply = llm.answer(question, hits)
    sources = sorted({h["source"] for h in hits if h["source"]})
    return {"answer": reply, "sources": sources}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Empty question")
    hits = rag.retrieve(question, namespace=req.session)
    sources = sorted({h["source"] for h in hits if h["source"]})

    def events():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        for piece in llm.answer_stream(question, hits):
            yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.delete("/reset")
def reset(session: str = DEFAULT_NS):
    rag.clear_namespace(session)
    return {"cleared": session}
