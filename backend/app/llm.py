from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL

_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You answer strictly from the provided context. "
    "Use only the information in the context to answer the question. "
    "If the answer is not in the context, reply that you don't know based on the uploaded documents. "
    "Do not use outside knowledge or make anything up. "
    "Answer in the same language as the question."
)

NO_CONTEXT_MSG = "I couldn't find anything about that in your uploaded documents."


def _messages(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return NO_CONTEXT_MSG
    resp = _client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=_messages(question, chunks),
    )
    return resp.choices[0].message.content


def answer_stream(question: str, chunks: list[dict]):
    if not chunks:
        yield NO_CONTEXT_MSG
        return
    stream = _client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=_messages(question, chunks),
        stream=True,
    )
    for chunk in stream:
        piece = chunk.choices[0].delta.content
        if piece:
            yield piece
