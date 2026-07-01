# DocChat

Upload a document, ask questions about it, get answers pulled only from that
document. If it's not in there, it says so instead of making something up.

Works in English, Georgian, and pretty much anything else, the embedding model
covers 89 languages.

## How it's put together

```
Next.js (Vercel) -> FastAPI (Render) -> Pinecone for vectors, Groq for answers
```

Upload a PDF/DOCX/CSV/TXT/MD file and the backend pulls the text out, chunks it,
and sends each chunk to Jina's embedding API. Those vectors land in Pinecone,
namespaced per browser session so nobody's docs mix with anyone else's.

When you ask something, the question gets embedded the same way, Pinecone
returns the closest chunks, and if nothing scores high enough it just tells you
it doesn't know rather than pretending. Otherwise the chunks go to Groq
(llama-3.3-70b) with a system prompt that keeps it locked to the provided
context. Answers stream back as they're generated.

## Running it yourself

Backend:

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in GROQ_API_KEY, PINECONE_API_KEY, JINA_API_KEY
uvicorn app.main:app --port 8000
```

Frontend:

```bash
cd frontend
npm install
cp .env.example .env.local   # BACKEND_URL=http://localhost:8000
npm run dev
```

localhost:3000, upload something, ask it about itself.

## Deploying

Backend on Render, set the root directory to `backend` (this repo has more
than one thing in it, Render won't find requirements.txt otherwise). Start
command:

```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Env vars: `GROQ_API_KEY`, `PINECONE_API_KEY`, `JINA_API_KEY`, and
`CORS_ORIGINS` pointed at wherever the frontend ends up living. Also set
`PYTHON_VERSION=3.12`, newer defaults have caused issues here before.

Frontend on Vercel, root directory `frontend`, one env var:
`BACKEND_URL` = your Render URL.

Embeddings run through Jina's API instead of loading a model locally,
originally tried bundling the model directly (fastembed + onnxruntime) but
that pushed Render's free tier over its 512MB limit on boot. Hosted embeddings
sidestep that entirely and keep the container small.

## Tuning it

`backend/app/config.py` has the knobs: `TOP_K` for how many chunks get
retrieved, `MIN_SCORE` for how relevant something has to be before it counts
(currently 0.30, anything below reads as "not actually related"), and
`CHUNK_SIZE`/`CHUNK_OVERLAP` for how documents get split up.
