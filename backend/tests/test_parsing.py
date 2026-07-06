import pytest

from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.parsing import chunk_blocks, extract_blocks, _pack, _units


def block(text, page=None):
    return {"text": text, "page": page}


def test_empty_input():
    assert chunk_blocks([]) == []
    assert chunk_blocks([block("   \n  ")]) == []


def test_short_text_single_chunk():
    chunks = chunk_blocks([block("One sentence. Another one here.")])
    assert len(chunks) == 1
    assert chunks[0]["text"] == "One sentence. Another one here."
    assert chunks[0]["page"] is None


def test_chunks_end_on_sentence_boundaries():
    sent = "This sentence has exactly seven words in it."
    text = " ".join([sent] * 100)
    chunks = chunk_blocks([block(text)])
    assert len(chunks) > 1
    for c in chunks:
        assert c["text"].endswith("."), "chunk cut mid-sentence"


def test_chunk_size_respected():
    sent = "Word " * 20
    units = _units(". ".join([sent.strip()] * 50) + ".")
    for text in _pack(units):
        assert len(text.split()) <= CHUNK_SIZE + CHUNK_OVERLAP


def test_overlap_carries_last_sentence():
    sents = [f"Sentence number {i} is right here filling space with words." for i in range(60)]
    out = _pack(sents)
    assert len(out) > 1
    # last sentence of chunk N reappears at the head of chunk N+1
    for prev, nxt in zip(out, out[1:]):
        tail = prev.rsplit(".", 2)[-2].strip() + "."
        assert nxt.startswith(tail)


def test_oversized_sentence_falls_back_to_word_split():
    giant = "word " * (CHUNK_SIZE * 2 + 10)
    units = _units(giant.strip())
    assert all(len(u.split()) <= CHUNK_SIZE for u in units)
    assert sum(len(u.split()) for u in units) == CHUNK_SIZE * 2 + 10


def test_csv_rows_never_split():
    header = "name,description\n"
    rows = "\n".join(f"item{i},{'detail ' * 30}" for i in range(40))
    blocks = extract_blocks("data.csv", (header + rows).encode())
    chunks = chunk_blocks(blocks)
    for c in chunks:
        # every row starts with "name: item" and must be intact within one chunk
        for part in c["text"].split("name: ")[1:]:
            assert "description:" in part


def test_pdf_page_provenance_kept():
    blocks = [block("First page text here.", page=1), block("Second page text here.", page=2)]
    chunks = chunk_blocks(blocks)
    assert [c["page"] for c in chunks] == [1, 2]


def test_no_overlap_only_trailing_chunk():
    sents = [f"Filler sentence number {i} with some extra words padding." for i in range(28)]
    out = _pack(sents)
    if len(out) > 1:
        assert out[-1] != out[-2].rsplit(".", 2)[-2].strip() + "."


def test_unsupported_extension():
    with pytest.raises(ValueError):
        extract_blocks("file.xyz", b"data")
