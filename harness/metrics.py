from dataclasses import dataclass, field

# Pricing per million tokens for claude-haiku-4-5
# Verify current rates at https://anthropic.com/pricing
HAIKU_INPUT_PRICE_PER_MTOK = 0.80
HAIKU_OUTPUT_PRICE_PER_MTOK = 4.00


@dataclass
class RunMetrics:
    results: list[dict] = field(default_factory=list)

    def add(self, result: dict) -> None:
        self.results.append(result)

    def summary(self) -> dict:
        n = len(self.results)
        if n == 0:
            return {}

        total_input = sum(r["total_input_tokens"] for r in self.results)
        total_output = sum(r["total_output_tokens"] for r in self.results)
        total_cost = (
            total_input / 1_000_000 * HAIKU_INPUT_PRICE_PER_MTOK
            + total_output / 1_000_000 * HAIKU_OUTPUT_PRICE_PER_MTOK
        )

        latencies = sorted(r["rag_latency_ms"] for r in self.results)
        relevance_scores = [r["context_relevance"]["score"] for r in self.results]
        correctness_scores = [r["correctness"]["score"] for r in self.results]
        faithful_count = sum(1 for r in self.results if r["faithfulness"]["faithful"])

        p95_index = max(0, int(n * 0.95) - 1)

        return {
            "n": n,
            "avg_context_relevance": round(sum(relevance_scores) / n, 2),
            "avg_correctness": round(sum(correctness_scores) / n, 2),
            "faithfulness_rate": round(faithful_count / n, 2),
            "avg_latency_ms": round(sum(latencies) / n, 1),
            "p95_latency_ms": round(latencies[p95_index], 1),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_usd": round(total_cost, 4),
            "cost_per_query_usd": round(total_cost / n, 5),
        }
