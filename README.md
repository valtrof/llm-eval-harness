# LLM Eval Harness

Automated evaluation framework for RAG (Retrieval-Augmented Generation) systems. Measures retrieval quality, answer faithfulness (hallucination detection), answer correctness, cost, and latency across a Q&A dataset using LLM-as-judge scoring.

## Why this exists

RAG systems fail in two places: the retriever returns irrelevant context, or the generator hallucinates facts not present in that context. Eyeballing outputs doesn't scale — and teams that ship RAG without an eval harness find out about regressions from their users. This harness runs structured, reproducible evaluations so retrieval quality and hallucination rates are measured on every change, the same way unit tests gate code.

**At a glance**

- Scores three failure modes separately: context relevance (retriever), faithfulness / hallucination (generator), and answer correctness (end-to-end)
- **100% faithfulness rate** across the benchmark, with per-query evidence for every judgment
- Full 12-query evaluation run costs **$0.0296** (~$0.0025/query) — cost and latency (avg/p95) tracked per query, so quality improvements have a price tag
- CI-ready: pytest suite fully mocks the Anthropic client — no API key or spend needed to run tests

## Architecture

```
Q&A Dataset (12 questions + ground truth answers)
        │
        ▼
TF-IDF Retriever ──→ top-3 relevant chunks from corpus
        │
        ▼
RAG Generator (Claude Haiku) ──→ grounded answer
        │
        ▼
Evaluator (Claude Haiku, LLM-as-judge)
  ├── Context Relevance  (0–3)  Is the retrieved context relevant to the question?
  ├── Faithfulness       (✓/✗)  Does the answer stay within the retrieved context?
  └── Answer Correctness (0–3)  Does the answer match the ground truth?
        │
        ▼
Report: per-query scores + aggregate summary (cost, latency, pass rates)
```

**Three evaluation dimensions:**

| Metric | Scale | What it catches |
|---|---|---|
| Context relevance | 0–3 | Retriever returning off-topic chunks |
| Faithfulness | ✓/✗ | Generator hallucinating facts not in context |
| Answer correctness | 0–3 | Correct context retrieved but wrong answer generated |

**Cost and latency tracking** per query and in aggregate, so you can measure the cost of improving quality.

## Stack

- Python 3.12
- [Anthropic Claude API](https://docs.anthropic.com) (Haiku for both generation and evaluation)
- scikit-learn TF-IDF, BM25 (`rank_bm25`), and `sentence-transformers` (bi-encoder + cross-encoder) retrieval
- `trafilatura` (HTML), `pdfplumber` (PDF text), `PyMuPDF` (PDF page rendering), and `easyocr` (scanned-PDF OCR) for real-document ingestion
- pytest with full mock coverage (no real API calls, no live network calls, in CI)
- GitHub Actions CI

## Quickstart

```bash
git clone https://github.com/valtrof/llm-eval-harness
cd llm-eval-harness
pip install -r requirements.txt
cp .env.example .env  # add your ANTHROPIC_API_KEY
python run_eval.py
```

## Example output

```
======================================================================
LLM EVAL HARNESS — PER-QUERY RESULTS
======================================================================

[q001] Why does Spark use lazy evaluation?
  Context relevance : 3/3  — Directly explains lazy evaluation and Catalyst optimizer.
  Faithfulness      : ✓  — All statements supported by retrieved context.
  Correctness       : 2/3  — Core concept correct; misses filter pushdown specifics.
  Latency           : 1981ms

[q002] What is the purpose of watermarks in Apache Flink?
  Context relevance : 3/3  — Directly covers watermark mechanism and BoundedOutOfOrderness.
  Faithfulness      : ✓  — Answer grounded in context.
  Correctness       : 3/3  — Accurate explanation of event-time progress and window closure.
  Latency           : 1434ms

[q005] What are the three conventional dbt model layers?
  Context relevance : 3/3  — Directly identifies all three layers with clear descriptions.
  Faithfulness      : ✓  — All information stated in context.
  Correctness       : 3/3  — Accurate and complete.
  Latency           : 997ms

[q010] What happens to Spark jobs when partitions are skewed?
  Context relevance : 3/3  — Explicitly addresses stragglers and job slowdown.
  Faithfulness      : ✓  — Directly quoted from context.
  Correctness       : 2/3  — Identifies stragglers but omits stage-wait bottleneck detail.
  Latency           : 1014ms

... (12 queries total)

======================================================================
AGGREGATE SUMMARY
======================================================================
  Samples               : 12
  Avg context relevance : 3.0/3
  Avg correctness       : 2.67/3
  Faithfulness rate     : 100%
  Avg latency           : 1512ms
  p95 latency           : 1951ms
  Total tokens          : 19,837 in / 3,429 out
  Total cost            : $0.0296
  Cost per query        : $0.00247
======================================================================
```

## Extending

**Swap the retriever** — replace `TfidfRetriever` with any retriever that implements `retrieve(query: str) -> list[dict]`. Drop in a vector store, BM25, or hybrid retriever without touching the evaluation logic.

**Swap the generator** — `RAGGenerator` wraps the Claude API; swap for any generator that returns `{"answer": str, "input_tokens": int, "output_tokens": int, "latency_ms": float}`.

**Add evaluation dimensions** — extend `Evaluator.evaluate()` with additional LLM-as-judge prompts (citation accuracy, answer completeness, toxicity checks).

**Add to your dataset** — `data/dataset.json` is a plain JSON array. Add Q&A pairs and the harness picks them up automatically.

## Document ingestion: real, messy PDF/HTML sources

`ingestion/` builds a corpus from real documents instead of hand-written text: 5 live official
documentation pages (Spark, Airflow, dbt, Flink, Snowflake — genuinely messy HTML with nav bars,
sidebars, and boilerplate) and 2 real academic papers (the Spark RDD paper and the Delta Lake
paper — genuinely messy two-column PDFs). `build_real_corpus.py` fetches, extracts, and chunks
all of them into `data/real_corpus.json`, separate from the hand-written `data/corpus.json` used
by the retrieval benchmarks above.

```bash
python build_real_corpus.py
```

**What it does:**
1. `ingestion/fetch.py` downloads each source once and caches the raw bytes under `data/raw/`
   (gitignored — regenerate with a fresh fetch, or delete the cache to force one).
2. `ingestion/html_extract.py` uses `trafilatura` to strip a documentation page down to its main
   content, discarding nav, sidebars, and cookie/boilerplate text a naive "grab all the text"
   extraction would keep.
3. `ingestion/pdf_extract.py` extracts body text from the two academic PDFs, fixing two specific
   problems naive extraction gets wrong on real two-column papers (see Findings), and checks every
   page for a real embedded text layer — any page with none (a scanned/image-based page) is
   rendered to an image and routed through `ingestion/ocr.py` instead of silently returning nothing
   for it.
4. `ingestion/chunker.py` packs extracted text into ~900-character passages along paragraph
   boundaries, with a 150-character overlap so a fact split across two paragraphs doesn't fall
   entirely on one side of a chunk boundary.

**Ingestion stats (live run):**

| Source | Type | Raw bytes | Extracted chars | Chunks |
|---|---|---|---|---|
| spark_docs | html | 167,093 | 57,563 | 91 |
| airflow_docs | html | 229,458 | 38,883 | 56 |
| dbt_docs | html | 165,268 | 6,041 | 9 |
| flink_docs | html | 104,268 | 8,469 | 12 |
| snowflake_docs | html | 752,147 | 9,322 | 15 |
| spark_paper | pdf | 886,315 | 70,876 | 100 |
| delta_lake_paper | pdf | 428,029 | 90,311 | 131 |
| ocr_demo_brochure | pdf (scanned) | 76,013 | 4,386 | 7 |
| ocr_demo_french | pdf (scanned) | 483,256 | 327 | 1 |
| **Total** | | **3,291,847** | **286,178** | **422** |

(HTML char/chunk counts shifted slightly from the previous run — extraction now outputs markdown
instead of plain text, specifically so real headings keep their `#` markers for section detection,
see below.)

**Findings:**
- **Two-column PDF reordering needed a real empty-gutter check, not a per-line heuristic.** The
  first version flagged single-column regions (title, abstract) as two-column whenever a line
  happened to end or start near the page's horizontal center, then split those lines in half and
  interleaved the halves — silently corrupting the most important part of the paper. The fix:
  project every word's bounding box onto the x-axis for the whole page, merge overlapping
  intervals, and look for an actual empty gap near the center. A real two-column gutter is empty
  across the *entire* page, not just on any one line; a single-column page has words scattered
  through the center on some lines even if a few individual lines don't reach it.
- **`pdfplumber`'s default word-tokenization tolerance merged entire lines into one "word."** Both
  academic PDFs encode inter-word spacing as small positioning gaps rather than explicit space
  characters, and the library's default `x_tolerance=3` was wider than that gap — e.g.
  `"justthatpartition.Thus,lostdatacanberecovered,often"` came back as a single token. Lowering
  `x_tolerance` to 1.5 fixed it; a value tuned empirically per-document, not a universal default,
  since a too-low tolerance would instead fracture real words apart.
- **HTML boilerplate stripping (`trafilatura`) worked without any manual tuning** — all 5 docs
  pages extracted cleanly on the first pass. The harder messiness in this pipeline turned out to
  be in the PDFs, not the HTML, which is the opposite of what "PDF/HTML ingestion" often implies
  going in.
- **Known remaining imperfections, left as-is rather than over-engineered:** hyphenated words
  split across a line break (`"applica-"` / `"tions"`) aren't rejoined, since detecting a genuine
  end-of-line hyphenation split vs. a real hyphenated word (`"fine-grained"`) reliably needs a
  dictionary lookup this project doesn't have. A paper's title-page copyright/footer box can still
  bleed into body text on that one page, since the general-purpose gutter detection isn't tuned
  for that specific box layout. Both are the kind of "good enough, documented, not silently wrong"
  tradeoffs a real ingestion pipeline has to make.

### OCR fallback for scanned/image-based PDFs

Neither of the two academic papers actually needed OCR — both have real embedded text layers. To
exercise the fallback path against genuine scans rather than a synthetic example, the corpus also
includes two real image-only PDFs with zero extractable text: a skewed scan of a 1980s MIDI
hardware product brochure and a French test paragraph, both public OCR test fixtures from the
[OCRmyPDF](https://github.com/ocrmypdf/OCRmyPDF) project (MIT licensed) — used here purely to
prove the fallback works, not for their (off-topic) content.

`ingestion/pdf_extract.py` checks every page's `pdfplumber` text extraction length before doing
anything else; a page with fewer than 20 characters of real text is assumed to be a scan, gets
rendered to a raster image via `PyMuPDF` at 300 DPI, and is OCR'd with `easyocr` (chosen over
wrapping the Tesseract binary because it's a pure Python/`pip install` — no external binary or
system PATH setup needed, since `torch` was already a dependency via `sentence-transformers`).
Pages with a real text layer never touch the OCR path at all, so the two academic papers' output
is byte-for-byte unaffected by this addition.

**Findings:**
- OCR correctly recovered real, mostly-legible text from both scans — e.g. the brochure's
  "LinnSequencer is a state-of-the-art composition and performance tool for the professional
  musician" came through cleanly, and the French paragraph's accented characters (é, î, ô, ç, œ)
  round-tripped correctly through UTF-8.
- OCR is genuinely lossy in a way the rest of this pipeline isn't: stylized title text came back
  garbled (`"32 Track MIDI H{iSqRgcc%e Recorder"` for a decorative all-caps heading), and
  ordinary body text has scattered word-level errors (`"Operation is similar"` → `"PoRwoRis
  snilar"`). This is the honest tradeoff of OCR vs. a real text layer — worth surfacing explicitly
  rather than presenting OCR'd chunks as equivalent-quality to the rest of the corpus.
- The chosen threshold (`extract_text()` returning fewer than 20 characters) is a text-layer
  presence check, not a quality check — it correctly separates "no embedded text at all" from
  "some embedded text," but doesn't detect the (rarer, harder) case of a PDF with a corrupted or
  partial text layer that still clears 20 characters. That's out of scope here; a production
  system would likely also sanity-check extracted text against the page's rendered content.

### Metadata extraction: titles, sections, dates

Every chunk in `data/real_corpus.json` also carries `source_title`, `source_date`, and `section` —
extracted from the actual document rather than hand-labeled. `ingestion/metadata.py` handles both
source types:

- **HTML** — `trafilatura.extract_metadata()` reads the page's `<title>`, OpenGraph tags, and the
  `htmldate` library's date-guessing heuristics. No custom parsing needed.
- **PDF title** — the embedded `Title` field is used when present, but both academic papers here
  leave it blank (LaTeX build tools set `Producer`/`Creator` but not a human-written `Title`), so
  the fallback takes the first line(s) of page one, stopping before a comma-heavy author list.
- **PDF date** — `CreationDate`, which *was* reliably present on both real papers (LaTeX sets it
  automatically), parsed from the PDF date format (`D:20120314225627-07'00'`) into an ISO date.
- **Sections** — `ingestion/chunker.py` tracks whichever heading most recently preceded each
  chunk's first paragraph while packing chunks, using two patterns: a markdown `#`-prefixed line
  (which is *why* `html_extract.py` outputs markdown instead of plain text — real HTML headings
  keep their marker) or a numbered section line like `"2.1 RDD Abstraction"` (the convention
  academic papers use).

**Metadata quality (live run):**

| Source | Title extracted | Date | Chunks with a section |
|---|---|---|---|
| spark_docs | "RDD Programming Guide" | (none) | 91/91 |
| airflow_docs | "Dags" | 2020-01-01 | 56/56 |
| dbt_docs | "SQL models \| dbt Developer Hub" | 2026-07-09 | 9/9 |
| flink_docs | "Timely Stream Processing" | 2015-12-04 | 12/12 |
| snowflake_docs | "Micro-partitions & Data Clustering" | 2026-01-01 | 15/15 |
| spark_paper | "Resilient Distributed Datasets: A Fault-Tolerant..." | 2012-03-14 | 89/100 |
| delta_lake_paper | "Delta Lake: High-Performance ACID Table Storage..." | 2020-08-21 | 96/131 |
| ocr_demo_brochure | (fallback to hardcoded source title) | 2016-01-19 | 6/7 |
| ocr_demo_french | "Adobe Photoshop PDF" | 2015-08-18 | 0/1 |

**Findings:**
- **The numbered-heading pattern, tuned for academic section numbering, false-positives on other
  digit-led capitalized text.** In the Spark paper, a numbered list item inside a section's body
  (`"3. Deserialization cost to convert binary records to us-"`) got misread as a new section
  boundary. In the OCR'd brochure, the product spec `"32 Track MDI"` (OCR's garbled read of "32
  Track MIDI", part of the actual page title) and the company address `"18720 Oxnard Street,
  Tarzana, CA 91356"` both matched the same regex, since "digits followed by capitalized words" is
  exactly what it looks for. This is a heuristic tuned for one document genre leaking into another
  — the fix would be genre-specific patterns, not a universal one.
- **`ocr_demo_brochure`'s title extraction fell all the way back to the hardcoded `Source.title`**,
  because the title fallback logic reads the first page's `pdfplumber` text layer — which is empty
  by definition for a scanned page (that's *why* it went through OCR). The OCR path and the title
  fallback path don't currently talk to each other; recovering a real title for a scanned document
  would mean running OCR before metadata extraction too, not just before chunking.
- **`ocr_demo_french`'s embedded `Title` field ("Adobe Photoshop PDF") is real metadata that's
  technically present but useless** — it's boilerplate the PDF-generating software wrote, not a
  human-authored title. The extraction logic has no way to distinguish a real title from
  software-generated placeholder text; it trusts embedded metadata when present, which is usually
  right but was wrong here.
- **Every HTML page's headings got tagged (91/91, 56/56, etc.) — full section coverage — while
  both PDFs only reached ~90%.** That gap is the front matter: the abstract and everything before
  a paper's first numbered section heading has no section to attach to, which is correct behavior
  (`section: null`), not a bug — there genuinely isn't a numbered section there yet.

## Running tests

```bash
pytest tests/ -v
```

Tests mock the Anthropic client — no API key required, no cost.

## Dataset

12 Q&A pairs covering Spark, Flink, Airflow, Kafka, BigQuery, dbt, Delta Lake, Snowflake, RAG, and
prompt caching, grounded in a 26-chunk corpus: 15 source-of-truth chunks plus 11 deliberate
lexical-overlap distractors (e.g. a Prefect chunk that name-drops Airflow's XComs, a Spark
Structured Streaming chunk that mentions Flink-style watermarking) added specifically to give
retrieval-quality comparisons real room to differ — see below. Each Q&A pair also carries a
`relevant_doc_ids` label used to score retrieval quality independently of the LLM-judge metrics.

## Retrieval comparison: TF-IDF vs. hybrid (BM25 + dense) + reranking

`run_eval_compare.py` runs the full harness twice over the same 12-question dataset — once with
the baseline `TfidfRetriever`, once with `rag/hybrid_retriever.py`'s `HybridRerankRetriever`
(BM25 + a dense bi-encoder fused via Reciprocal Rank Fusion, then reranked by a cross-encoder) —
and reports retrieval-quality and LLM-judged metrics side by side. It's a live run: two full
generation + LLM-as-judge passes over the dataset, so it costs roughly 2x a single `run_eval.py`
run (~$0.06 total in the run below).

```bash
python run_eval_compare.py
```

**Results (top_k=3, live run):**

| Metric | TF-IDF | Hybrid + rerank |
|---|---|---|
| Retrieval recall@3 | 1.00 | 1.00 |
| Retrieval top-1 accuracy | 0.50 | 0.92 |
| Avg context relevance | 3.0/3 | 3.0/3 |
| Faithfulness rate | 100% | 100% |
| Avg correctness | 2.67/3 | 2.75/3 |
| Avg RAG latency | 1892ms | 1696ms |
| Total cost | $0.0285 | $0.0297 |

**Findings:**
- **Recall@3 was saturated at 1.00 for both retrievers** — with `top_k=3` on a 26-chunk corpus,
  the correct chunk almost always ended up somewhere in the top 3 for both methods, even when it
  wasn't ranked first. Recall@k alone would say "no difference here," which would be the wrong
  conclusion.
- **Top-1 accuracy told the real story: 0.50 → 0.92.** TF-IDF's pure term-frequency scoring got
  fooled by 6 of 12 distractor chunks into the #1 slot (e.g. ranking the Redshift-sort-keys
  distractor above the actual BigQuery clustering chunk for a BigQuery question); hybrid+rerank
  only missed 1. This is the retrieval-ranking-quality gain the reranker and hybrid-retrieval
  experiments in `curriculum-engine` predicted, now measured against a downstream generator
  instead of a synthetic recall/nDCG label.
- **That ranking gain barely moved end-to-end LLM-judged quality** (faithfulness stayed 100%,
  correctness moved 2.67→2.75, well within judge noise) **because `top_k=3` already gave Claude a
  safety net** — the correct chunk was present in the context either way, just not always first.
  The honest conclusion: at a generous top_k, a weak retriever's ranking mistakes get absorbed by
  the generator reading past them. Retrieval ranking quality would matter far more at `top_k=1`,
  under a tighter context/cost budget, or on a corpus where the correct chunk sometimes falls
  outside the top_k entirely — this dataset's recall@3 saturation means that failure mode isn't
  exercised here. Latency and cost differences between conditions are within normal run-to-run
  variance for live Claude Haiku calls, not attributable to the retrieval change.
