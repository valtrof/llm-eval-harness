"""Orchestrates fetch -> extract -> chunk for every configured source, producing corpus.json-
shaped records plus per-source extraction stats and document metadata."""

import io
from dataclasses import dataclass

import pdfplumber

from ingestion.chunker import chunk_with_sections
from ingestion.fetch import fetch
from ingestion.html_extract import extract_html
from ingestion.metadata import extract_html_metadata, extract_pdf_metadata
from ingestion.pdf_extract import extract_pdf
from ingestion.sources import Source


@dataclass
class SourceResult:
    source: Source
    raw_bytes: int
    extracted_chars: int
    chunks: list[dict]  # [{"text": ..., "section": ... | None}, ...]
    title: str | None
    date: str | None


def process_source(source: Source) -> SourceResult:
    raw = fetch(source)

    if source.doc_type == "html":
        text = extract_html(raw)
        meta = extract_html_metadata(raw)
    else:
        text = extract_pdf(raw)
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            meta = extract_pdf_metadata(pdf)

    return SourceResult(
        source=source,
        raw_bytes=len(raw),
        extracted_chars=len(text),
        chunks=chunk_with_sections(text),
        title=meta["title"] or source.title,
        date=meta["date"],
    )


def build_corpus(results: list[SourceResult]) -> list[dict]:
    """Flatten per-source chunks into a corpus.json-compatible list of {id, topic, text, ...}."""
    corpus = []
    counter = 0
    for result in results:
        for chunk in result.chunks:
            counter += 1
            corpus.append(
                {
                    "id": f"r{counter:03d}",
                    "topic": result.source.topic,
                    "text": chunk["text"],
                    "section": chunk["section"],
                    "source_id": result.source.id,
                    "source_title": result.title,
                    "source_date": result.date,
                    "source_url": result.source.url,
                }
            )
    return corpus
