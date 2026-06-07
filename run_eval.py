#!/usr/bin/env python3
"""
LLM Eval Harness — evaluates a RAG system across a dataset of Q&A pairs.

Usage:
    python run_eval.py
    python run_eval.py --dataset data/dataset.json --corpus data/corpus.json --output eval_report.json

Requires: ANTHROPIC_API_KEY environment variable
"""
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from harness.dataset import load_dataset
from harness.evaluator import Evaluator
from harness.metrics import RunMetrics
from harness.report import print_report, save_report
from rag.generator import RAGGenerator
from rag.retriever import TfidfRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM eval harness")
    parser.add_argument("--dataset", default="data/dataset.json")
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument("--output", default="eval_report.json")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve per query")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Error: ANTHROPIC_API_KEY is not set")

    print(f"Loading dataset from {args.dataset}...")
    samples = load_dataset(args.dataset)
    print(f"Loaded {len(samples)} samples\n")

    retriever = TfidfRetriever(corpus_path=args.corpus, top_k=args.top_k)
    generator = RAGGenerator()
    evaluator = Evaluator()
    metrics = RunMetrics()

    results = []
    for i, sample in enumerate(samples, 1):
        q_display = (sample.question[:50] + "...") if len(sample.question) > 53 else sample.question
        print(f"[{i}/{len(samples)}] {sample.id}: {q_display}")

        chunks = retriever.retrieve(sample.question)
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
            "ground_truth": sample.ground_truth_answer,
            "rag_latency_ms": gen["latency_ms"],
            "rag_input_tokens": gen["input_tokens"],
            "rag_output_tokens": gen["output_tokens"],
            **eval_result,
            "total_input_tokens": gen["input_tokens"] + eval_result["eval_input_tokens"],
            "total_output_tokens": gen["output_tokens"] + eval_result["eval_output_tokens"],
        }
        results.append(result)
        metrics.add(result)

    summary = metrics.summary()
    print_report(results, summary)
    save_report(results, summary, args.output)


if __name__ == "__main__":
    main()
