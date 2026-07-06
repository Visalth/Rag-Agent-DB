import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
JINA_API_KEY = os.environ["JINA_API_KEY"]
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-docs-chat")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
JINA_MODEL = os.getenv("JINA_MODEL", "jina-embeddings-v5-text-small")
RERANK_MODEL = os.getenv("RERANK_MODEL", "jina-reranker-v3")

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

EMBED_DIM = 384
TOP_K = 5           # chunks that reach the LLM after reranking
RETRIEVE_K = 15     # candidates pulled from pinecone before dedup/rerank
MIN_SCORE = 0.20    # absolute noise floor: direct questions score ~0.45-0.66, unrelated noise ~0.08-0.16
RELATIVE_CUTOFF = 0.35   # also drop hits below this fraction of the best hit's score
DEDUP_JACCARD = 0.8      # word-set similarity above which two chunks count as duplicates
CHUNK_SIZE = 250      # words per chunk, kept small so several chunks fit groq's request limit
CHUNK_OVERLAP = 50    # max words carried over from the previous chunk's last sentence
