import time
import uuid
import requests
from pinecone import Pinecone, ServerlessSpec
from pinecone.errors.exceptions import NotFoundError

from .config import (
    PINECONE_API_KEY, PINECONE_INDEX, EMBED_DIM, TOP_K, MIN_SCORE,
    JINA_API_KEY, JINA_MODEL,
)

JINA_URL = "https://api.jina.ai/v1/embeddings"

_pc = Pinecone(api_key=PINECONE_API_KEY)

if not _pc.has_index(PINECONE_INDEX):
    _pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # wait until the new index is ready before using it
    for _ in range(30):
        if _pc.describe_index(PINECONE_INDEX).status.get("ready"):
            break
        time.sleep(1)

_index = _pc.Index(PINECONE_INDEX)


def _embed(texts: list[str], task: str) -> list[list[float]]:
    resp = requests.post(
        JINA_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
        },
        json={
            "model": JINA_MODEL,
            "task": task,
            "normalized": True,
            "output_dimension": EMBED_DIM,
            "input": texts,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


def embed_passages(chunks: list[str]) -> list[list[float]]:
    return _embed(chunks, task="retrieval.passage")


def embed_query(question: str) -> list[float]:
    return _embed([question], task="retrieval.query")[0]


def store_chunks(chunks: list[str], source: str, namespace: str) -> int:
    if not chunks:
        return 0
    vectors = []
    for text, values in zip(chunks, embed_passages(chunks)):
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": values,
            "metadata": {"text": text, "source": source},
        })
    # upsert in batches to stay under request limits
    for i in range(0, len(vectors), 100):
        _index.upsert(vectors=vectors[i:i + 100], namespace=namespace)
    return len(vectors)


def retrieve(question: str, namespace: str) -> list[dict]:
    res = _index.query(
        vector=embed_query(question),
        top_k=TOP_K,
        include_metadata=True,
        namespace=namespace,
    )
    out = []
    for m in res.get("matches", []):
        if m.get("score", 0.0) < MIN_SCORE:
            continue
        md = m.get("metadata") or {}
        out.append({"text": md.get("text", ""), "source": md.get("source", ""), "score": m.get("score", 0.0)})
    return out


def clear_namespace(namespace: str) -> None:
    try:
        _index.delete(delete_all=True, namespace=namespace)
    except NotFoundError:
        pass  # nothing stored for this session yet
