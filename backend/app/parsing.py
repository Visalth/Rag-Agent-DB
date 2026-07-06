import csv
import io
import re
from pypdf import PdfReader
from docx import Document

from .config import CHUNK_SIZE, CHUNK_OVERLAP

# sentence boundary: ., !, ?, … followed by whitespace; newlines are hard boundaries
_SENT_RE = re.compile(r"(?<=[.!?…])\s+")


def extract_blocks(filename: str, data: bytes) -> list[dict]:
    """Returns [{text, page}] where page is set only for PDFs."""
    name = filename.lower()
    if name.endswith(".pdf"):
        return _pdf(data)
    if name.endswith(".docx"):
        return _one_block(_docx(data))
    if name.endswith(".csv"):
        return _one_block(_csv(data))
    if name.endswith((".txt", ".md")):
        return _one_block(data.decode("utf-8", errors="ignore"))
    raise ValueError(f"Unsupported file type: {filename}")


def _one_block(text: str) -> list[dict]:
    return [{"text": text, "page": None}] if text.strip() else []


def _pdf(data: bytes) -> list[dict]:
    reader = PdfReader(io.BytesIO(data))
    blocks = []
    for n, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            blocks.append({"text": text, "page": n})
    return blocks


def _docx(data: bytes) -> str:
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs)


def _csv(data: bytes) -> str:
    text = data.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        return ""
    header = rows[0]
    lines = []
    for row in rows[1:]:
        pairs = [f"{h}: {v}" for h, v in zip(header, row)]
        lines.append(", ".join(pairs))
    return "\n".join(lines)


def chunk_blocks(blocks: list[dict]) -> list[dict]:
    """Sentence-aware chunking. Chunks never cross block (pdf page) boundaries,
    so page provenance stays accurate. Returns [{text, page}]."""
    chunks = []
    for block in blocks:
        for text in _pack(_units(block["text"])):
            chunks.append({"text": text, "page": block["page"]})
    return chunks


def _units(text: str) -> list[str]:
    """Split into sentences. Lines (csv rows, paragraphs) are hard boundaries,
    so a csv row is never cut in half. Oversized sentences fall back to word splits."""
    units = []
    for line in text.splitlines():
        for sent in _SENT_RE.split(line.strip()):
            sent = sent.strip()
            if not sent:
                continue
            words = sent.split()
            if len(words) > CHUNK_SIZE:
                for i in range(0, len(words), CHUNK_SIZE):
                    units.append(" ".join(words[i:i + CHUNK_SIZE]))
            else:
                units.append(sent)
    return units


def _pack(units: list[str]) -> list[str]:
    """Greedily pack sentences into chunks of up to CHUNK_SIZE words, carrying the
    previous chunk's last sentence as overlap when it fits in CHUNK_OVERLAP words."""
    out = []
    cur: list[str] = []
    cur_words = 0
    for unit in units:
        w = len(unit.split())
        if cur and cur_words + w > CHUNK_SIZE:
            out.append(" ".join(cur))
            tail = cur[-1]
            tail_w = len(tail.split())
            if tail_w <= CHUNK_OVERLAP:
                cur, cur_words = [tail], tail_w
            else:
                cur, cur_words = [], 0
        cur.append(unit)
        cur_words += w
    if cur and not (out and len(cur) == 1 and out[-1].endswith(cur[0])):
        out.append(" ".join(cur))
    return out
