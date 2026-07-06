import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("PINECONE_API_KEY", "test")
os.environ.setdefault("JINA_API_KEY", "test")
