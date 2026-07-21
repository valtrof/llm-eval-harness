import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalSample:
    id: str
    question: str
    ground_truth_answer: str
    relevant_doc_ids: list[str] = field(default_factory=list)


def load_dataset(path: str = "data/dataset.json") -> list[EvalSample]:
    with open(path) as f:
        records = json.load(f)
    return [EvalSample(**r) for r in records]
