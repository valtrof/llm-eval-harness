"""Document-level metadata (title, date) and section-heading detection, so ingested chunks carry
more than just raw text — the same title/date/section fields a hand-curated corpus would have.
"""

import re
from datetime import date

import trafilatura

_PDF_DATE_RE = re.compile(r"D:(\d{4})(\d{2})(\d{2})")
_PDF_NUMBERED_HEADING_RE = re.compile(r"^[0-9]+(\.[0-9]+)*\.?\s+[A-Z][A-Za-z0-9 ,'\-]{2,60}$")
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$")


def extract_html_metadata(raw_html: bytes) -> dict:
    """Title/publish-date via trafilatura's own metadata extractor (uses page <title>, OpenGraph
    tags, and the htmldate library's date-guessing heuristics) — no hand-written parsing needed."""
    meta = trafilatura.extract_metadata(raw_html)
    if meta is None:
        return {"title": None, "date": None}
    return {"title": meta.title or None, "date": meta.date or None}


def _parse_pdf_date(raw_date: str | None) -> str | None:
    if not raw_date:
        return None
    match = _PDF_DATE_RE.match(raw_date)
    if not match:
        return None
    year, month, day = (int(g) for g in match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _guess_pdf_title(first_page_text: str) -> str | None:
    """Academic PDFs built from LaTeX rarely set the embedded Title metadata field, so fall back
    to the page's first line(s) — a paper's title is reliably the first text on page one, often
    wrapped across two lines before the author list (which is comma-heavy) starts."""
    lines = [l.strip() for l in first_page_text.splitlines() if l.strip()]
    if not lines:
        return None
    title_lines = [lines[0]]
    if len(lines) > 1 and "," not in lines[1] and len(lines[1]) < 80 and lines[1][:1].isupper():
        title_lines.append(lines[1])
    return " ".join(title_lines)


def extract_pdf_metadata(pdf) -> dict:
    """pdf is an open pdfplumber.PDF. Embedded Title metadata is trusted when present; CreationDate
    (present on both real PDFs tested here, since LaTeX build tools set it automatically) is
    parsed into an ISO date."""
    info = pdf.metadata or {}
    title = (info.get("Title") or "").strip() or None
    date_str = _parse_pdf_date(info.get("CreationDate")) or _parse_pdf_date(info.get("ModDate"))

    if not title and pdf.pages:
        first_page_text = pdf.pages[0].extract_text() or ""
        title = _guess_pdf_title(first_page_text)

    return {"title": title, "date": date_str}


def parse_heading(line: str) -> str | None:
    """Return the cleaned heading text if `line` looks like a section heading, else None.
    Two independent patterns, one per source type: a markdown '#'-prefixed line (from HTML pages,
    extracted as markdown specifically so real headings keep this marker) or a numbered section
    line like '2.1 RDD Abstraction' (the convention academic papers use)."""
    line = line.strip()
    if not line:
        return None

    markdown_match = _MARKDOWN_HEADING_RE.match(line)
    if markdown_match:
        return markdown_match.group(1).strip()

    if _PDF_NUMBERED_HEADING_RE.match(line):
        return line

    return None
