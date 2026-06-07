import time
import anthropic


class RAGGenerator:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self._client = anthropic.Anthropic()
        self._model = model

    def generate(self, question: str, context_chunks: list[dict]) -> dict:
        context = "\n\n".join(chunk["text"] for chunk in context_chunks)
        prompt = (
            "Answer the question using ONLY the information in the provided context.\n"
            "If the context does not contain enough information to answer, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )

        start = time.perf_counter()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "answer": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_ms": latency_ms,
            "model": self._model,
        }
