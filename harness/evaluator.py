import json
import anthropic

CONTEXT_RELEVANCE_PROMPT = """\
Rate the relevance of the retrieved context to the question on a scale of 0-3.

Question: {question}
Context: {context}

0 = Completely irrelevant
1 = Slightly relevant but mostly off-topic
2 = Moderately relevant, contains some useful information
3 = Highly relevant, directly addresses the question

Respond with JSON only: {{"score": <0-3>, "reason": "<one sentence>"}}"""

FAITHFULNESS_PROMPT = """\
Does the answer contain ONLY information that can be found in or directly inferred from the context?

Context: {context}
Answer: {answer}

Respond with JSON only: {{"faithful": true/false, "reason": "<one sentence noting any hallucinated claims>"}}"""

CORRECTNESS_PROMPT = """\
Rate the correctness of the AI answer against the ground truth on a scale of 0-3.

Question: {question}
Ground truth: {ground_truth}
AI answer: {answer}

0 = Completely wrong or irrelevant
1 = Partially correct but missing key information or contains errors
2 = Mostly correct with minor gaps or inaccuracies
3 = Correct and complete

Respond with JSON only: {{"score": <0-3>, "reason": "<one sentence>"}}"""


class Evaluator:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self._client = anthropic.Anthropic()
        self._model = model

    def _judge(self, prompt: str) -> tuple[dict, object]:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text), response.usage

    def evaluate(
        self,
        question: str,
        context: str,
        answer: str,
        ground_truth: str,
    ) -> dict:
        total_input_tokens = 0
        total_output_tokens = 0

        relevance_result, usage = self._judge(
            CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context)
        )
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens

        faithfulness_result, usage = self._judge(
            FAITHFULNESS_PROMPT.format(context=context, answer=answer)
        )
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens

        correctness_result, usage = self._judge(
            CORRECTNESS_PROMPT.format(
                question=question, ground_truth=ground_truth, answer=answer
            )
        )
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens

        return {
            "context_relevance": relevance_result,
            "faithfulness": faithfulness_result,
            "correctness": correctness_result,
            "eval_input_tokens": total_input_tokens,
            "eval_output_tokens": total_output_tokens,
        }
