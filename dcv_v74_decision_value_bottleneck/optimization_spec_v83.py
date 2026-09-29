"""Shared, framework-independent optimization task definition."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


METRIC_NAMES = (
    "collision",
    "non_traversable",
    "safety_risk",
    "route_deviation",
    "lack_of_progress",
    "path_length",
    "discomfort",
    "goal_error",
)


@dataclass
class OptimizationTask:
    name: str
    domain: str
    weights: dict
    limits: dict
    constraint_penalty: float
    explanation: str

    @property
    def weight_vector(self):
        return [float(self.weights[name]) for name in METRIC_NAMES]

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
