from groq import Groq
from .config import GROQ_API_KEY, GROQ_MODEL
_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You answer strictly from the provided context. "
    "Use only the information in the context to answer the question. "
    "If the context is empty or doesn't contain the answer, say in your own words "
    "that you don't know, based on the documents uploaded so far - vary your phrasing "
    "naturally instead of repeating a fixed sentence. "
    "Do not use outside knowledge or make anything up. "
    "Answer in the same language as the question."
)


# groq free tier rejects requests over 12k tokens, and georgian text runs
# close to 1 token per character, so keep the context well under that
MAX_CONTEXT_CHARS = 8000


def _messages(question: str, chunks: list[dict]) -> list[dict]:
    parts = []
    used = 0
    for c in chunks:
        block = f"[Source: {c['source']}]\n{c['text']}"
        if parts and used + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        used += len(block)
    if parts:
        context = "\n\n---\n\n".join(parts)
    else:
        context = "(no matching content found in any uploaded document)"
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def answer(question: str, chunks: list[dict]) -> str:
    resp = _client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=_messages(question, chunks),
    )
    return resp.choices[0].message.content


def answer_stream(question: str, chunks: list[dict]):
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
