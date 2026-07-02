import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
JINA_API_KEY = os.environ["JINA_API_KEY"]
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-docs-chat")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
JINA_MODEL = os.getenv("JINA_MODEL", "jina-embeddings-v5-text-small")

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

EMBED_DIM = 384
TOP_K = 5
MIN_SCORE = 0.20   # direct questions score ~0.45-0.66, vague ones ("what's this doc about") ~0.28, unrelated noise ~0.08-0.16
CHUNK_SIZE = 250      # words per chunk, kept small so several chunks fit groq's request limit
CHUNK_OVERLAP = 50
