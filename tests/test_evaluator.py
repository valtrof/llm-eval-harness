from unittest.mock import MagicMock, patch

from harness.evaluator import Evaluator


def _mock_response(content: str):
    msg = MagicMock()
    msg.content = [MagicMock(text=content)]
    msg.usage = MagicMock(input_tokens=100, output_tokens=20)
    return msg


@patch("harness.evaluator.anthropic.Anthropic")
def test_evaluate_returns_all_dimensions(mock_anthropic_cls):
    client = MagicMock()
    mock_anthropic_cls.return_value = client
    client.messages.create.side_effect = [
        _mock_response('{"score": 3, "reason": "Directly relevant."}'),
        _mock_response('{"faithful": true, "reason": "All claims supported."}'),
        _mock_response('{"score": 2, "reason": "Mostly correct."}'),
    ]

    result = Evaluator().evaluate(
        question="What is lazy evaluation in Spark?",
        context="Spark uses lazy evaluation to optimize plans before execution.",
        answer="Spark defers execution until an action is called.",
        ground_truth="Spark uses lazy evaluation to build a logical plan before executing.",
    )

    assert result["context_relevance"]["score"] == 3
    assert result["faithfulness"]["faithful"] is True
    assert result["correctness"]["score"] == 2
    assert result["eval_input_tokens"] == 300
    assert result["eval_output_tokens"] == 60


@patch("harness.evaluator.anthropic.Anthropic")
def test_evaluate_detects_unfaithful_answer(mock_anthropic_cls):
    client = MagicMock()
    mock_anthropic_cls.return_value = client
    client.messages.create.side_effect = [
        _mock_response('{"score": 2, "reason": "Somewhat relevant."}'),
        _mock_response('{"faithful": false, "reason": "Answer mentions Catalyst optimizer which is not in context."}'),
        _mock_response('{"score": 1, "reason": "Partially correct."}'),
    ]

    result = Evaluator().evaluate(
        question="How does Spark work?",
        context="Spark is a distributed computation engine.",
        answer="Spark uses the Catalyst optimizer to build execution plans.",
        ground_truth="Spark is a distributed computation engine.",
    )

    assert result["faithfulness"]["faithful"] is False
    assert "Catalyst" in result["faithfulness"]["reason"]
