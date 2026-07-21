from unittest.mock import MagicMock, patch

from ingestion.chunker import chunk_text, chunk_with_sections
from ingestion.html_extract import extract_html
from ingestion.metadata import _guess_pdf_title, _parse_pdf_date, extract_pdf_metadata, parse_heading
from ingestion.pdf_extract import (
    _find_gutter_x,
    _has_text_layer,
    _normalize_for_boilerplate,
    _strip_repeated_boilerplate,
    extract_pdf,
)
from ingestion.sources import Source


def _word(text: str, x0: float, x1: float, top: float) -> dict:
    return {"text": text, "x0": x0, "x1": x1, "top": top, "bottom": top + 10}


def test_chunk_text_packs_paragraphs_up_to_target_size():
    paragraphs = ["word " * 20 for _ in range(5)]  # ~100 chars each
    text = "\n".join(paragraphs)

    chunks = chunk_text(text, target_chars=250, overlap_chars=20)

    assert len(chunks) > 1
    assert all(len(c) <= 250 + 20 + 5 for c in chunks)  # allow slack for overlap + one paragraph


def test_chunk_text_carries_overlap_between_chunks():
    text = "\n".join(f"Paragraph {i} " + ("x" * 80) for i in range(5))

    chunks = chunk_text(text, target_chars=150, overlap_chars=30)

    assert len(chunks) > 1
    # the tail of chunk N should reappear at the start of chunk N+1
    tail = chunks[0][-30:]
    assert tail in chunks[1]


def test_chunk_text_splits_oversized_single_paragraph():
    huge_paragraph = "word " * 400  # ~2000 chars, no newlines at all
    chunks = chunk_text(huge_paragraph, target_chars=300, overlap_chars=0)

    assert len(chunks) > 1
    assert all(len(c) <= 310 for c in chunks)


def test_chunk_text_empty_input_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n   ") == []


def test_chunk_with_sections_tags_chunks_with_preceding_markdown_heading():
    text = "\n".join(
        [
            "# Overview",
            "Some intro text before any subsection heading appears here at all.",
            "# Linking with Spark",
            "Details about linking go here, well past the minimum useful length.",
        ]
    )

    chunks = chunk_with_sections(text, target_chars=100, overlap_chars=0)

    sections = [c["section"] for c in chunks]
    assert "Overview" in sections
    assert "Linking with Spark" in sections
    # the chunk under "Overview" shouldn't be tagged with the later heading
    overview_chunk = next(c for c in chunks if c["section"] == "Overview")
    assert "intro text" in overview_chunk["text"]


def test_chunk_with_sections_numbered_pdf_heading():
    text = "\n".join(
        [
            "Some abstract-like text with no section heading above it at all here.",
            "2.1 RDD Abstraction",
            "Body text describing the RDD abstraction in detail follows this heading.",
        ]
    )

    chunks = chunk_with_sections(text, target_chars=60, overlap_chars=0)

    assert chunks[0]["section"] is None
    assert any(c["section"] == "2.1 RDD Abstraction" for c in chunks)


def test_chunk_text_still_returns_plain_strings():
    text = "# Heading\nSome body text under the heading, long enough to matter here."
    chunks = chunk_text(text)
    assert all(isinstance(c, str) for c in chunks)


def test_parse_heading_detects_markdown_heading():
    assert parse_heading("## Linking with Spark") == "Linking with Spark"
    assert parse_heading("# Overview") == "Overview"


def test_parse_heading_detects_numbered_pdf_section():
    assert parse_heading("2.1 RDD Abstraction") == "2.1 RDD Abstraction"
    assert parse_heading("9 Conclusion") == "9 Conclusion"


def test_parse_heading_rejects_ordinary_body_text():
    assert parse_heading("This is just a normal sentence, not a heading at all.") is None
    assert parse_heading("") is None


def test_parse_pdf_date_parses_pdf_date_format():
    assert _parse_pdf_date("D:20120314225627-07'00'") == "2012-03-14"


def test_parse_pdf_date_returns_none_for_missing_or_malformed():
    assert _parse_pdf_date(None) is None
    assert _parse_pdf_date("not a date") is None


def test_guess_pdf_title_takes_first_lines_before_author_list():
    first_page = "Resilient Distributed Datasets: A Fault-Tolerant Abstraction for\nIn-Memory Cluster Computing\nMatei Zaharia, Mosharaf Chowdhury, ..."
    title = _guess_pdf_title(first_page)
    assert title == "Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing"


def test_guess_pdf_title_stops_before_comma_heavy_author_line():
    first_page = "A Short Paper Title\nAlice Smith, Bob Jones, Carol Lee"
    title = _guess_pdf_title(first_page)
    assert title == "A Short Paper Title"


def test_guess_pdf_title_returns_none_for_empty_page():
    assert _guess_pdf_title("") is None


def test_extract_pdf_metadata_prefers_embedded_title():
    mock_pdf = MagicMock()
    mock_pdf.metadata = {"Title": "  A Real Embedded Title  ", "CreationDate": "D:20200821101401-07'00'"}
    mock_pdf.pages = []

    meta = extract_pdf_metadata(mock_pdf)

    assert meta["title"] == "A Real Embedded Title"
    assert meta["date"] == "2020-08-21"


def test_extract_pdf_metadata_falls_back_to_first_page_when_title_missing():
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "A Paper With No Embedded Title\nAuthor One, Author Two"

    mock_pdf = MagicMock()
    mock_pdf.metadata = {"Title": ""}
    mock_pdf.pages = [mock_page]

    meta = extract_pdf_metadata(mock_pdf)

    assert meta["title"] == "A Paper With No Embedded Title"


def test_html_extract_drops_nav_and_keeps_main_content():
    raw_html = b"""
    <html><body>
      <nav><a href="/">Home</a><a href="/docs">Docs</a><a href="/blog">Blog</a></nav>
      <header>Site Header Banner</header>
      <main>
        <h1>Understanding Lazy Evaluation</h1>
        <p>Apache Spark defers execution of transformations until an action is called,
        allowing the Catalyst optimizer to analyze the complete logical plan first and
        reorder operations for efficiency before anything actually runs on the cluster.</p>
      </main>
      <footer>Copyright 2026 Example Corp. All rights reserved. Privacy Policy | Terms</footer>
    </body></html>
    """

    text = extract_html(raw_html)

    assert "Catalyst optimizer" in text
    assert "lazy evaluation" in text.lower() or "Lazy Evaluation" in text
    assert "Privacy Policy" not in text
    assert "Copyright 2026 Example Corp" not in text


def test_html_extract_returns_empty_string_on_unparseable_input():
    assert extract_html(b"") == ""


def test_find_gutter_x_detects_real_two_column_gap():
    # two clear columns: left words end by x=200, right words start at x=260, page width 400
    words = [_word(f"l{i}", 100, 190, top=10.0 * i) for i in range(10)]
    words += [_word(f"r{i}", 260, 350, top=10.0 * i) for i in range(10)]

    gutter = _find_gutter_x(words, page_width=400)

    assert gutter is not None
    assert 190 < gutter < 260


def test_find_gutter_x_returns_none_for_single_column_page():
    # words spread evenly across the full width with no consistent empty gap
    words = [_word(f"w{i}", x0, x0 + 30, top=10.0 * (i % 5)) for i, x0 in enumerate(range(50, 350, 15))]

    gutter = _find_gutter_x(words, page_width=400)

    assert gutter is None


def test_normalize_for_boilerplate_collapses_page_numbers():
    assert _normalize_for_boilerplate("Page 3") == _normalize_for_boilerplate("Page 47")


def test_strip_repeated_boilerplate_removes_running_header():
    pages_lines = [
        ["Running Paper Title", "Some real body text for page one.", "3"],
        ["Running Paper Title", "Different body text on page two here.", "4"],
        ["Running Paper Title", "Yet more distinct content on page three.", "5"],
        ["Running Paper Title", "And finally page four's unique content.", "6"],
    ]

    cleaned = _strip_repeated_boilerplate(pages_lines)

    for page in cleaned:
        assert "Running Paper Title" not in page
        assert not any(line.strip().isdigit() for line in page)
    assert "Some real body text for page one." in cleaned[0]


def test_strip_repeated_boilerplate_keeps_short_page_lists_untouched():
    pages_lines = [["Header"], ["Header"]]
    assert _strip_repeated_boilerplate(pages_lines) == pages_lines


def test_has_text_layer_true_for_real_text():
    page = MagicMock()
    page.extract_text.return_value = "This page has plenty of real, extractable body text on it."
    assert _has_text_layer(page) is True


def test_has_text_layer_false_for_scanned_page():
    page = MagicMock()
    page.extract_text.return_value = ""
    assert _has_text_layer(page) is False


def test_has_text_layer_false_for_stray_artifact_chars():
    page = MagicMock()
    page.extract_text.return_value = "  3  "  # a bare page number, not a real text layer
    assert _has_text_layer(page) is False


@patch("ingestion.pdf_extract.ocr_page")
@patch("ingestion.pdf_extract.fitz.open")
@patch("ingestion.pdf_extract.pdfplumber.open")
def test_extract_pdf_routes_textless_page_to_ocr(mock_pdfplumber_open, mock_fitz_open, mock_ocr_page):
    text_page = MagicMock()
    text_page.extract_text.return_value = "A real paragraph of body text on this page, plenty long."
    text_page.extract_words.return_value = [
        _word("Real", 72, 100, 10.0),
        _word("text.", 102, 130, 10.0),
    ]
    text_page.width = 400

    scanned_page = MagicMock()
    scanned_page.extract_text.return_value = ""

    mock_pdf = MagicMock()
    mock_pdf.pages = [text_page, scanned_page]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

    mock_fitz_doc = MagicMock()
    mock_fitz_open.return_value = mock_fitz_doc
    mock_ocr_page.return_value = ["OCR'd line one.", "OCR'd line two."]

    result = extract_pdf(b"fake pdf bytes")

    mock_fitz_open.assert_called_once()
    mock_ocr_page.assert_called_once_with(mock_fitz_doc[1])
    assert "Real text." in result
    assert "OCR'd line one." in result
    assert "OCR'd line two." in result


@patch("ingestion.pdf_extract.fitz.open")
@patch("ingestion.pdf_extract.pdfplumber.open")
def test_extract_pdf_skips_ocr_entirely_when_all_pages_have_text(mock_pdfplumber_open, mock_fitz_open):
    text_page = MagicMock()
    text_page.extract_text.return_value = "Plenty of real body text right here, easily long enough."
    text_page.extract_words.return_value = [_word("Hello", 72, 110, 10.0)]
    text_page.width = 400

    mock_pdf = MagicMock()
    mock_pdf.pages = [text_page]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

    extract_pdf(b"fake pdf bytes")

    mock_fitz_open.assert_not_called()


@patch("easyocr.Reader")
def test_ocr_page_reuses_cached_reader_across_calls(mock_reader_cls):
    import ingestion.ocr as ocr_module
    from ingestion.ocr import ocr_page

    ocr_module._READER = None  # reset the module-level singleton for this test
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = ["Line one", "Line two"]
    mock_reader_cls.return_value = mock_reader

    fake_page = MagicMock()
    fake_pix = MagicMock(height=10, width=10, n=3, samples=bytes(10 * 10 * 3))
    fake_page.get_pixmap.return_value = fake_pix

    result1 = ocr_page(fake_page)
    result2 = ocr_page(fake_page)

    assert result1 == ["Line one", "Line two"]
    assert result2 == ["Line one", "Line two"]
    mock_reader_cls.assert_called_once()  # Reader constructed once, reused on the second call
    ocr_module._READER = None  # leave the singleton clean for other tests


@patch("ingestion.fetch.requests.get")
def test_fetch_uses_cache_when_present(mock_get, tmp_path):
    from ingestion.fetch import fetch

    source = Source(id="test_doc", topic="test", doc_type="html", url="https://example.com", title="Test")
    cached_path = tmp_path / "test_doc.html"
    cached_path.write_bytes(b"cached content")

    result = fetch(source, raw_dir=tmp_path)

    assert result == b"cached content"
    mock_get.assert_not_called()


@patch("ingestion.fetch.requests.get")
def test_fetch_downloads_and_caches_when_absent(mock_get, tmp_path):
    from ingestion.fetch import fetch

    mock_response = MagicMock(content=b"downloaded content")
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    source = Source(id="test_doc", topic="test", doc_type="html", url="https://example.com", title="Test")
    result = fetch(source, raw_dir=tmp_path)

    assert result == b"downloaded content"
    assert (tmp_path / "test_doc.html").read_bytes() == b"downloaded content"
    mock_get.assert_called_once()
