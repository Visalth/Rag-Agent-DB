import hashlib
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pinecone import Pinecone, ServerlessSpec
from pinecone.errors.exceptions import NotFoundError

from .config import (
    PINECONE_API_KEY, PINECONE_INDEX, EMBED_DIM, TOP_K, RETRIEVE_K,
    MIN_SCORE, RELATIVE_CUTOFF, DEDUP_JACCARD,
    JINA_API_KEY, JINA_MODEL, RERANK_MODEL,
)

JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"

_index = None
_http = None


def _session() -> requests.Session:
    global _http
    if _http is None:
        s = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503],
            allowed_methods=["POST"],
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        _http = s
    return _http


def _get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if not pc.has_index(PINECONE_INDEX):
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # wait until the new index is ready before using it
            for _ in range(30):
                if pc.describe_index(PINECONE_INDEX).status.get("ready"):
                    break
                time.sleep(1)
        _index = pc.Index(PINECONE_INDEX)
    return _index


def _jina(url: str, payload: dict) -> dict:
    resp = _session().post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _embed(texts: list[str], task: str) -> list[list[float]]:
    data = _jina(JINA_EMBED_URL, {
        "model": JINA_MODEL,
        "task": task,
        "normalized": True,
        "output_dimension": EMBED_DIM,
        "input": texts,
    })["data"]
    return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


def embed_passages(chunks: list[str]) -> list[list[float]]:
    return _embed(chunks, task="retrieval.passage")


def embed_query(question: str) -> list[float]:
    return _embed([question], task="retrieval.query")[0]


def source_prefix(source: str) -> str:
    return hashlib.sha1(source.encode()).hexdigest()[:16]


def delete_source(source: str, namespace: str) -> None:
    """Remove all vectors previously stored for this file, so re-uploads
    replace instead of accumulate."""
    index = _get_index()
    try:
        ids = []
        for page in index.list(prefix=source_prefix(source) + "#", namespace=namespace):
            ids.extend(v.id for v in page.vectors)
        for i in range(0, len(ids), 100):
            index.delete(ids=ids[i:i + 100], namespace=namespace)
    except NotFoundError:
        pass  # namespace doesn't exist yet


def store_chunks(chunks: list[dict], source: str, namespace: str) -> int:
    if not chunks:
        return 0
    delete_source(source, namespace)
    prefix = source_prefix(source)
    vectors = []
    embeddings = embed_passages([c["text"] for c in chunks])
    for i, (chunk, values) in enumerate(zip(chunks, embeddings)):
        md = {"text": chunk["text"], "source": source}
        if chunk.get("page") is not None:
            md["page"] = chunk["page"]
        vectors.append({"id": f"{prefix}#{i}", "values": values, "metadata": md})
    # upsert in batches to stay under request limits
    index = _get_index()
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i + 100], namespace=namespace)
    return len(vectors)


def _filter_hits(matches: list[dict]) -> list[dict]:
    hits = []
    for m in matches:
        score = m.get("score", 0.0)
        if score < MIN_SCORE:
            continue
        md = m.get("metadata") or {}
        hits.append({
            "text": md.get("text", ""),
            "source": md.get("source", ""),
            "page": int(md["page"]) if md.get("page") is not None else None,
            "score": score,
        })
    if hits:
        floor = hits[0]["score"] * RELATIVE_CUTOFF
        hits = [h for h in hits if h["score"] >= floor]
    return hits


def _dedup(hits: list[dict]) -> list[dict]:
    """Drop near-duplicate chunks (overlap twins, leftover copies), keeping the
    higher-scored one. Hits arrive sorted by score."""
    kept = []
    for h in hits:
        words = set(h["text"].lower().split())
        dup = False
        for k in kept:
            kw = k["_words"]
            union = len(words | kw)
            if union and len(words & kw) / union >= DEDUP_JACCARD:
                dup = True
                break
        if not dup:
            kept.append({**h, "_words": words})
    return [{k: v for k, v in h.items() if k != "_words"} for h in kept]


def _rerank(question: str, hits: list[dict]) -> list[dict]:
    """Reorder hits by cross-encoder relevance. Non-fatal: falls back to
    cosine order if the rerank call fails."""
    if len(hits) <= 1:
        return hits[:TOP_K]
    try:
        data = _jina(JINA_RERANK_URL, {
            "model": RERANK_MODEL,
            "query": question,
            "documents": [h["text"] for h in hits],
            "top_n": TOP_K,
            "return_documents": False,
        })
        return [hits[r["index"]] for r in data["results"]]
    except (requests.RequestException, KeyError, IndexError):
        return hits[:TOP_K]


def retrieve(question: str, namespace: str, source: str | None = None) -> list[dict]:
    index = _get_index()
    res = index.query(
        vector=embed_query(question),
        top_k=RETRIEVE_K,
        include_metadata=True,
        namespace=namespace,
        filter={"source": {"$eq": source}} if source else None,
    )
    hits = _filter_hits(res.get("matches", []))
    hits = _dedup(hits)
    return _rerank(question, hits)


def clear_namespace(namespace: str) -> None:
    try:
        _get_index().delete(delete_all=True, namespace=namespace)
    except NotFoundError:
        pass  # nothing stored for this session yet
