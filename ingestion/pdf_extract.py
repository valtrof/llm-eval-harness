"""Extracts body text from real academic-paper PDFs, which naive "extract all text" gets wrong
in two specific ways this module fixes: two-column layout interleaves both columns line-by-line
instead of reading the left column fully before the right, and every page repeats a running
header/footer (title, page number) that shouldn't appear in the middle of the extracted body text.
"""

import io
import re

import fitz  # PyMuPDF — only used to render pages that need OCR
import pdfplumber

from ingestion.ocr import ocr_page

_PAGE_NUM_RE = re.compile(r"^\s*\d+\s*$")
TEXT_LAYER_MIN_CHARS = 20


def _cluster_lines(words: list[dict], tolerance: float = 3.0) -> list[list[dict]]:
    """Group words into lines by similar vertical position; each line's words are left-to-right."""
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: w["top"]):
        for line in lines:
            if abs(line[0]["top"] - w["top"]) <= tolerance:
                line.append(w)
                break
        else:
            lines.append([w])
    lines.sort(key=lambda line: line[0]["top"])
    for line in lines:
        line.sort(key=lambda w: w["x0"])
    return lines


def _line_text(line: list[dict]) -> str:
    return " ".join(w["text"] for w in line)


def _find_gutter_x(words: list[dict], page_width: float, min_gap: float = 8.0) -> float | None:
    """Find a real column gutter by projecting every word's [x0, x1] box onto the x-axis, merging
    overlaps, and looking for an empty gap near the page center. This is robust to text simply
    running close to the center on a single-column page (which per-line "does any word touch the
    middle band" checks falsely flag as two columns whenever a line happens to end or start near
    the middle) — a real gutter is empty across the *whole* page, not just on any one line."""
    if not words:
        return None
    intervals = sorted((w["x0"], w["x1"]) for w in words)
    merged: list[list[float]] = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    mid = page_width / 2
    search_band = page_width * 0.25
    best: tuple[float, float] | None = None  # (gap_center, gap_width)
    for (_, a_end), (b_start, _) in zip(merged, merged[1:]):
        gap_width = b_start - a_end
        gap_center = (a_end + b_start) / 2
        if gap_width >= min_gap and abs(gap_center - mid) <= search_band:
            if best is None or gap_width > best[1]:
                best = (gap_center, gap_width)
    return best[0] if best else None


def _page_lines(page) -> list[str]:
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False, x_tolerance=1.5)
    if not words:
        return []

    gutter_x = _find_gutter_x(words, page.width)
    left = sum(1 for w in words if w["x1"] <= gutter_x) if gutter_x else 0
    right = sum(1 for w in words if w["x0"] >= gutter_x) if gutter_x else 0

    if gutter_x is not None and left >= 5 and right >= 5:
        column_groups = [
            [w for w in words if w["x1"] <= gutter_x],
            [w for w in words if w["x0"] >= gutter_x],
        ]
    else:
        column_groups = [words]

    lines = []
    for group in column_groups:
        lines.extend(_line_text(line) for line in _cluster_lines(group))
    return lines


def _normalize_for_boilerplate(line: str) -> str:
    """Collapse digits so a running header like 'Page 3' and 'Page 12' count as the same line."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _strip_repeated_boilerplate(pages_lines: list[list[str]]) -> list[list[str]]:
    """Drop lines repeating near-verbatim across most pages (running headers/footers) and bare
    page numbers."""
    if len(pages_lines) < 3:
        return pages_lines

    counts: dict[str, int] = {}
    for lines in pages_lines:
        for key in {_normalize_for_boilerplate(l) for l in lines if l.strip()}:
            counts[key] = counts.get(key, 0) + 1

    threshold = max(3, int(len(pages_lines) * 0.5))
    boilerplate_keys = {key for key, count in counts.items() if count >= threshold}

    cleaned = []
    for lines in pages_lines:
        kept = [
            l
            for l in lines
            if l.strip()
            and not _PAGE_NUM_RE.match(l)
            and _normalize_for_boilerplate(l) not in boilerplate_keys
        ]
        cleaned.append(kept)
    return cleaned


def _has_text_layer(page, min_chars: int = TEXT_LAYER_MIN_CHARS) -> bool:
    """A scanned/image-only page has no embedded text at all (or a handful of stray artifact
    characters) — a real text layer clears this trivially, so a low threshold is enough to tell
    the two apart without false-positiving on a nearly-blank real page."""
    text = page.extract_text() or ""
    return len(text.strip()) >= min_chars


def extract_pdf(raw_pdf: bytes) -> str:
    """Return the PDF's body text, with multi-column pages reordered into correct reading order,
    repeated running headers/footers/page numbers stripped, and any page that has no text layer
    at all (a scanned/image-based page) routed through OCR instead of returning nothing for it."""
    with pdfplumber.open(io.BytesIO(raw_pdf)) as pdf:
        needs_ocr = [not _has_text_layer(page) for page in pdf.pages]
        fitz_doc = fitz.open(stream=raw_pdf, filetype="pdf") if any(needs_ocr) else None

        pages_lines = []
        for i, page in enumerate(pdf.pages):
            if needs_ocr[i]:
                pages_lines.append(ocr_page(fitz_doc[i]))
            else:
                pages_lines.append(_page_lines(page))

        if fitz_doc is not None:
            fitz_doc.close()

    pages_lines = _strip_repeated_boilerplate(pages_lines)
    pages_text = ["\n".join(lines) for lines in pages_lines if lines]
    return "\n\n".join(pages_text)
