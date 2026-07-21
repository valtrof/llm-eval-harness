"""Splits extracted document text into retrieval-sized passages, packing whole paragraphs up to
a target size rather than cutting mid-sentence, with a small overlap so a fact split across two
paragraphs doesn't fall entirely on one side of a chunk boundary. Each chunk also carries the
section heading (if any) that precedes it in the source document."""

from ingestion.metadata import parse_heading

TARGET_CHARS = 900
OVERLAP_CHARS = 150


def _split_oversized(paragraph: str, target_chars: int) -> list[str]:
    """A paragraph longer than target_chars on its own (e.g. a block trafilatura didn't split
    further) still needs to be broken up, so it doesn't become a single oversized chunk."""
    words = paragraph.split()
    pieces: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + len(word) + 1 > target_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces


def chunk_with_sections(
    text: str, target_chars: int = TARGET_CHARS, overlap_chars: int = OVERLAP_CHARS
) -> list[dict]:
    """Return [{"text": ..., "section": ...}, ...]. `section` is whichever heading most recently
    preceded a chunk's first paragraph, or None if the chunk starts before any heading."""
    raw_paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not raw_paragraphs:
        return []

    current_section: str | None = None
    tagged_paragraphs: list[tuple[str, str | None]] = []
    for paragraph in raw_paragraphs:
        heading = parse_heading(paragraph)
        if heading:
            current_section = heading
        pieces = _split_oversized(paragraph, target_chars) if len(paragraph) > target_chars else [paragraph]
        tagged_paragraphs.extend((piece, current_section) for piece in pieces)

    chunks: list[dict] = []
    current_text = ""
    current_chunk_section: str | None = None

    for paragraph, section in tagged_paragraphs:
        if current_text and len(current_text) + len(paragraph) + 2 > target_chars:
            chunks.append({"text": current_text.strip(), "section": current_chunk_section})
            tail = current_text[-overlap_chars:] if overlap_chars > 0 else ""
            current_text = f"{tail}\n\n{paragraph}" if tail else paragraph
            current_chunk_section = section
        else:
            if not current_text:
                current_chunk_section = section
            current_text = f"{current_text}\n\n{paragraph}" if current_text else paragraph

    if current_text.strip():
        chunks.append({"text": current_text.strip(), "section": current_chunk_section})

    return chunks


def chunk_text(text: str, target_chars: int = TARGET_CHARS, overlap_chars: int = OVERLAP_CHARS) -> list[str]:
    return [c["text"] for c in chunk_with_sections(text, target_chars, overlap_chars)]
