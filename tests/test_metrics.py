from harness.metrics import RunMetrics


def _make_result(correctness=3, relevance=3, faithful=True, latency_ms=500.0, in_tokens=200, out_tokens=100):
    return {
        "context_relevance": {"score": relevance, "reason": ""},
        "faithfulness": {"faithful": faithful, "reason": ""},
        "correctness": {"score": correctness, "reason": ""},
        "rag_latency_ms": latency_ms,
        "total_input_tokens": in_tokens,
        "total_output_tokens": out_tokens,
    }


def test_summary_aggregates_correctly():
    metrics = RunMetrics()
    metrics.add(_make_result(correctness=3, relevance=3, faithful=True, latency_ms=400))
    metrics.add(_make_result(correctness=1, relevance=2, faithful=False, latency_ms=600))

    s = metrics.summary()
    assert s["n"] == 2
    assert s["avg_correctness"] == 2.0
    assert s["avg_context_relevance"] == 2.5
    assert s["faithfulness_rate"] == 0.5
    assert s["avg_latency_ms"] == 500.0


def test_cost_calculation():
    metrics = RunMetrics()
    metrics.add(_make_result(in_tokens=1_000_000, out_tokens=1_000_000))

    s = metrics.summary()
    # 1M input @ $0.80/MTok + 1M output @ $4.00/MTok = $4.80
    assert s["total_cost_usd"] == 4.8


def test_empty_metrics_returns_empty():
    assert RunMetrics().summary() == {}
