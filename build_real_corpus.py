#!/usr/bin/env python3
"""
Ingests real, messy source documents (official docs HTML pages + academic-paper PDFs) into a
corpus.json-shaped output, exercising the ingestion pipeline against real-world documents rather
than the hand-written data/corpus.json used by the retrieval benchmarks.

Usage:
    python build_real_corpus.py
    python build_real_corpus.py --output data/real_corpus.json --force-refetch
"""
import argparse
import json

from ingestion.pipeline import SourceResult, build_corpus, process_source
from ingestion.sources import SOURCES


def print_stats(results: list[SourceResult]) -> None:
    print("\n" + "=" * 78)
    print("INGESTION STATS")
    print("=" * 78)
    print(f"{'source':<16} {'type':<5} {'raw bytes':>10} {'extracted chars':>16} {'chunks':>7}")
    for r in results:
        print(
            f"{r.source.id:<16} {r.source.doc_type:<5} {r.raw_bytes:>10,} "
            f"{r.extracted_chars:>16,} {len(r.chunks):>7}"
        )
    total_raw = sum(r.raw_bytes for r in results)
    total_chars = sum(r.extracted_chars for r in results)
    total_chunks = sum(len(r.chunks) for r in results)
    print("-" * 78)
    print(f"{'TOTAL':<16} {'':<5} {total_raw:>10,} {total_chars:>16,} {total_chunks:>7}")
    print("=" * 78 + "\n")

    print("=" * 100)
    print("METADATA")
    print("=" * 100)
    print(f"{'source':<18} {'w/ section':>10} {'title':<50} {'date':>10}")
    for r in results:
        with_section = sum(1 for c in r.chunks if c["section"])
        title = (r.title or "(none)")[:48]
        print(f"{r.source.id:<18} {with_section:>3}/{len(r.chunks):<6} {title:<50} {r.date or '(none)':>10}")
    total_with_section = sum(1 for r in results for c in r.chunks if c["section"])
    print("-" * 100)
    print(f"{'TOTAL chunks with a section':<40} {total_with_section}/{total_chunks}")
    print("=" * 100 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real-document corpus via the ingestion pipeline")
    parser.add_argument("--output", default="data/real_corpus.json")
    args = parser.parse_args()

    results = []
    for source in SOURCES:
        print(f"Processing {source.id} ({source.doc_type}) — {source.url}")
        results.append(process_source(source))

    print_stats(results)

    corpus = build_corpus(results)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)
    print(f"Wrote {len(corpus)} chunks to {args.output}")


if __name__ == "__main__":
    main()
