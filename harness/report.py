import json


def print_report(results: list[dict], summary: dict) -> None:
    print("\n" + "=" * 70)
    print("LLM EVAL HARNESS — PER-QUERY RESULTS")
    print("=" * 70)

    for r in results:
        q = r["question"]
        q_display = (q[:57] + "...") if len(q) > 60 else q
        faithful_icon = "✓" if r["faithfulness"]["faithful"] else "✗"
        print(f"\n[{r['id']}] {q_display}")
        print(f"  Context relevance : {r['context_relevance']['score']}/3  — {r['context_relevance']['reason']}")
        print(f"  Faithfulness      : {faithful_icon}  — {r['faithfulness']['reason']}")
        print(f"  Correctness       : {r['correctness']['score']}/3  — {r['correctness']['reason']}")
        print(f"  Latency           : {r['rag_latency_ms']:.0f}ms")

    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)
    print(f"  Samples               : {summary['n']}")
    print(f"  Avg context relevance : {summary['avg_context_relevance']}/3")
    print(f"  Avg correctness       : {summary['avg_correctness']}/3")
    print(f"  Faithfulness rate     : {summary['faithfulness_rate']:.0%}")
    print(f"  Avg latency           : {summary['avg_latency_ms']}ms")
    print(f"  p95 latency           : {summary['p95_latency_ms']}ms")
    print(f"  Total tokens          : {summary['total_input_tokens']:,} in / {summary['total_output_tokens']:,} out")
    print(f"  Total cost            : ${summary['total_cost_usd']:.4f}")
    print(f"  Cost per query        : ${summary['cost_per_query_usd']:.5f}")
    print("=" * 70 + "\n")


def save_report(results: list[dict], summary: dict, path: str = "eval_report.json") -> None:
    with open(path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"Full report saved to {path}")
