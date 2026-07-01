import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-docs-chat")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

EMBED_DIM = 384
TOP_K = 5
MIN_SCORE = 0.30   # calibrated: real matches ~0.44-0.51, unrelated noise ~0.08-0.16
CHUNK_SIZE = 600      # words per chunk
CHUNK_OVERLAP = 100
