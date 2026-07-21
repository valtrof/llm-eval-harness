"""Fetches source documents and caches the raw bytes on disk, so extraction can be iterated on
and tested without re-hitting the network every run."""

from pathlib import Path

import requests

from ingestion.sources import Source

RAW_DIR = Path("data/raw")
TIMEOUT = 20
USER_AGENT = "llm-eval-harness-ingestion/1.0 (portfolio project; contact via github.com/valtrof)"


def _raw_path(source: Source) -> Path:
    ext = "html" if source.doc_type == "html" else "pdf"
    return RAW_DIR / f"{source.id}.{ext}"


def fetch(source: Source, raw_dir: Path = RAW_DIR, force: bool = False) -> bytes:
    """Return the raw document bytes, using the on-disk cache unless force=True."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / _raw_path(source).name

    if path.exists() and not force:
        return path.read_bytes()

    response = requests.get(source.url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    response.raise_for_status()
    path.write_bytes(response.content)
    return response.content
