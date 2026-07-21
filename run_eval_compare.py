#!/usr/bin/env python3
"""
Before/after comparison: TF-IDF-only retrieval vs. BM25+dense hybrid retrieval with
cross-encoder reranking, both run through the full LLM-judged eval harness.

Usage:
    python run_eval_compare.py
    python run_eval_compare.py --dataset data/dataset.json --corpus data/corpus.json

Requires: ANTHROPIC_API_KEY environment variable. Makes live Anthropic API calls for
generation and LLM-as-judge evaluation, once per sample per retriever (2x the cost of
run_eval.py for the same dataset).
"""
import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from harness.dataset import EvalSample, load_dataset
from harness.evaluator import Evaluator
from harness.metrics import RunMetrics
from rag.generator import RAGGenerator
from rag.hybrid_retriever import HybridRerankRetriever
from rag.retriever import TfidfRetriever


def run_condition(
    name: str,
    retriever,
    samples: list[EvalSample],
    generator: RAGGenerator,
    evaluator: Evaluator,
) -> tuple[list[dict], dict]:
    metrics = RunMetrics()
    results = []
    retrieval_hits = 0
    retrieval_top1_hits = 0

    for i, sample in enumerate(samples, 1):
        print(f"  [{name}] [{i}/{len(samples)}] {sample.id}")

        chunks = retriever.retrieve(sample.question)
        retrieved_ids = [c["id"] for c in chunks]
        relevant = set(sample.relevant_doc_ids)
        if relevant & set(retrieved_ids):
            retrieval_hits += 1
        if retrieved_ids and retrieved_ids[0] in relevant:
            retrieval_top1_hits += 1

        context = "\n\n".join(c["text"] for c in chunks)
        gen = generator.generate(sample.question, chunks)
        eval_result = evaluator.evaluate(
            question=sample.question,
            context=context,
            answer=gen["answer"],
            ground_truth=sample.ground_truth_answer,
        )

        result = {
            "id": sample.id,
            "question": sample.question,
            "answer": gen["answer"],
            "retrieved_ids": retrieved_ids,
            "relevant_doc_ids": sample.relevant_doc_ids,
            "rag_latency_ms": gen["latency_ms"],
            **eval_result,
            "total_input_tokens": gen["input_tokens"] + eval_result["eval_input_tokens"],
            "total_output_tokens": gen["output_tokens"] + eval_result["eval_output_tokens"],
        }
        results.append(result)
        metrics.add(result)

    summary = metrics.summary()
    summary["retrieval_recall_at_k"] = round(retrieval_hits / len(samples), 2)
    summary["retrieval_top1_accuracy"] = round(retrieval_top1_hits / len(samples), 2)
    return results, summary


def print_comparison(top_k: int, tfidf_summary: dict, hybrid_summary: dict) -> None:
    rows = [
        ("retrieval_recall_at_k", f"Retrieval recall@{top_k}"),
        ("retrieval_top1_accuracy", "Retrieval top-1 accuracy"),
        ("avg_context_relevance", "Avg context relevance"),
        ("faithfulness_rate", "Faithfulness rate"),
        ("avg_correctness", "Avg correctness"),
        ("avg_latency_ms", "Avg RAG latency (ms)"),
        ("total_cost_usd", "Total cost (USD)"),
    ]
    print("\n" + "=" * 70)
    print("BEFORE/AFTER COMPARISON — TF-IDF vs hybrid (BM25+dense) + rerank")
    print("=" * 70)
    print(f"{'metric':<26} {'tfidf':>14} {'hybrid+rerank':>16}")
    for key, label in rows:
        print(f"{label:<26} {tfidf_summary[key]:>14} {hybrid_summary[key]:>16}")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TF-IDF vs hybrid+rerank retrieval")
    parser.add_argument("--dataset", default="data/dataset.json")
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument("--output", default="eval_report_compare.json")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve per query")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Error: ANTHROPIC_API_KEY is not set")

    print(f"Loading dataset from {args.dataset}...")
    samples = load_dataset(args.dataset)
    print(f"Loaded {len(samples)} samples\n")

    generator = RAGGenerator()
    evaluator = Evaluator()

    print("=== Baseline: TF-IDF retrieval ===")
    tfidf_retriever = TfidfRetriever(corpus_path=args.corpus, top_k=args.top_k)
    tfidf_results, tfidf_summary = run_condition(
        "tfidf", tfidf_retriever, samples, generator, evaluator
    )

    print("\n=== Hybrid (BM25 + dense, RRF-fused) + cross-encoder rerank ===")
    hybrid_retriever = HybridRerankRetriever(corpus_path=args.corpus, top_k=args.top_k)
    hybrid_results, hybrid_summary = run_condition(
        "hybrid", hybrid_retriever, samples, generator, evaluator
    )

    print_comparison(args.top_k, tfidf_summary, hybrid_summary)

    with open(args.output, "w") as f:
        json.dump(
            {
                "tfidf": {"summary": tfidf_summary, "results": tfidf_results},
                "hybrid_rerank": {"summary": hybrid_summary, "results": hybrid_results},
            },
            f,
            indent=2,
        )
    print(f"Full comparison saved to {args.output}")


if __name__ == "__main__":
    main()
