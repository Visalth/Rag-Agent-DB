from app.llm import MAX_CONTEXT_CHARS, _messages, source_label


def hit(text, source="doc.pdf", page=None, score=0.5):
    return {"text": text, "source": source, "page": page, "score": score}


def test_empty_hits_message():
    msgs = _messages("q", [])
    assert "(no matching content found" in msgs[1]["content"]


def test_page_in_source_label():
    assert source_label(hit("t", page=4)) == "doc.pdf (p. 4)"
    assert source_label(hit("t")) == "doc.pdf"
    assert "[Source: doc.pdf (p. 4)]" in _messages("q", [hit("t", page=4)])[1]["content"]


def test_truncation_keeps_top_ranked():
    big = "x" * (MAX_CONTEXT_CHARS - 20)
    hits = [hit(big), hit("second best"), hit("third")]
    content = _messages("q", hits)[1]["content"]
    assert big in content
    assert "second best" not in content
    assert "third" not in content


def test_first_chunk_always_included():
    huge = "x" * (MAX_CONTEXT_CHARS * 2)
    content = _messages("q", [hit(huge)])[1]["content"]
    assert huge in content
