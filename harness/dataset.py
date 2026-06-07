import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalSample:
    id: str
    question: str
    ground_truth_answer: str


def load_dataset(path: str = "data/dataset.json") -> list[EvalSample]:
    with open(path) as f:
        records = json.load(f)
    return [EvalSample(**r) for r in records]
