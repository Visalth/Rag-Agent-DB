# DocChat

Upload your documents and ask questions about them. Answers come only from what you
upload, if it's not in your files, DocChat says so instead of guessing.

Multilingual: works in English, Georgian, and 50+ other languages.

## How it works

```
Next.js (Vercel)  ->  FastAPI (Render)  ->  Pinecone (vectors)
   chat UI              parse + embed          Groq (answers)
```

1. You upload a PDF, DOCX, CSV, TXT, or MD file.
2. The backend extracts the text, splits it into chunks, and embeds each chunk with
   a multilingual model (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim).
3. Chunks go into Pinecone, scoped to your session so your files stay private.
4. When you ask a question, the most relevant chunks are retrieved and passed to
   Groq (`llama-3.3-70b-versatile`) with a strict prompt: answer only from this
   context. Weakly-related matches are dropped, so off-topic questions get refused.

Answers stream back token by token.

## Run locally

Backend:

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # fill in GROQ_API_KEY and PINECONE_API_KEY
uvicorn app.main:app --port 8000
```

Frontend:

```bash
cd frontend
npm install
cp .env.example .env.local   # BACKEND_URL=http://localhost:8000
npm run dev
```

Open http://localhost:3000.

## Deploy

**Backend on Render** (web service):
- Root directory: `backend`
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Env: `GROQ_API_KEY`, `PINECONE_API_KEY`, and `CORS_ORIGINS` set to your Vercel URL
- Pin Python to 3.12 (Render env var `PYTHON_VERSION=3.12`)
- The vector store is Pinecone (hosted), so Render's ephemeral disk is fine

**Frontend on Vercel**:
- Root directory: `frontend`
- Env: `BACKEND_URL` set to your Render URL

## Config

Tunable in `backend/app/config.py`: `TOP_K`, `MIN_SCORE` (retrieval cutoff),
`CHUNK_SIZE`, `CHUNK_OVERLAP`.
