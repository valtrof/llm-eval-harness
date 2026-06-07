# LLM Eval Harness

Automated evaluation framework for RAG (Retrieval-Augmented Generation) systems. Measures retrieval quality, answer faithfulness (hallucination detection), answer correctness, cost, and latency across a Q&A dataset using LLM-as-judge scoring.

## Why this exists

RAG systems fail in two places: the retriever returns irrelevant context, or the generator hallucinates facts not present in that context. Eyeballing outputs doesn't scale. This harness runs structured, reproducible evaluations so regressions are caught before they reach production.

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
- scikit-learn TF-IDF for retrieval (no embedding API required)
- pytest with full mock coverage (no real API calls in CI)
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
  Context relevance : 3/3  — Directly addresses lazy evaluation and Catalyst optimizer.
  Faithfulness      : ✓  — All claims present in retrieved context.
  Correctness       : 3/3  — Correct and complete.
  Latency           : 847ms

[q002] What is the purpose of watermarks in Apache Flink?
  Context relevance : 3/3  — Flink watermark chunk retrieved correctly.
  Faithfulness      : ✓  — Answer grounded in context.
  Correctness       : 3/3  — Accurate explanation of BoundedOutOfOrderness.
  Latency           : 923ms

...

======================================================================
AGGREGATE SUMMARY
======================================================================
  Samples               : 12
  Avg context relevance : 2.8/3
  Avg correctness       : 2.7/3
  Faithfulness rate     : 100%
  Avg latency           : 910ms
  p95 latency           : 1240ms
  Total tokens          : 48,320 in / 6,840 out
  Total cost            : $0.0661
  Cost per query        : $0.00551
======================================================================
```

## Extending

**Swap the retriever** — replace `TfidfRetriever` with any retriever that implements `retrieve(query: str) -> list[dict]`. Drop in a vector store, BM25, or hybrid retriever without touching the evaluation logic.

**Swap the generator** — `RAGGenerator` wraps the Claude API; swap for any generator that returns `{"answer": str, "input_tokens": int, "output_tokens": int, "latency_ms": float}`.

**Add evaluation dimensions** — extend `Evaluator.evaluate()` with additional LLM-as-judge prompts (citation accuracy, answer completeness, toxicity checks).

**Add to your dataset** — `data/dataset.json` is a plain JSON array. Add Q&A pairs and the harness picks them up automatically.

## Running tests

```bash
pytest tests/ -v
```

Tests mock the Anthropic client — no API key required, no cost.

## Dataset

12 Q&A pairs covering Spark, Flink, Airflow, Kafka, BigQuery, dbt, Delta Lake, Snowflake, RAG, and prompt caching — all grounded in a 15-chunk public corpus of data engineering concepts.
