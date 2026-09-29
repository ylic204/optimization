import unittest

import numpy as np

from candidate_milp_solver_v83 import solve_candidate_milp
from mapped_optimization_schema_v83 import make_sample
from optimization_spec_v83 import METRIC_NAMES, OptimizationTask


class OptimizationPipelineTest(unittest.TestCase):
    def test_schema_and_milp(self):
        reference = np.stack(
            [np.linspace(0, 10, 20), np.zeros(20)], axis=-1
        ).astype(np.float32)
        metrics = {
            name: np.full(11, 0.2, dtype=np.float32) for name in METRIC_NAMES
        }
        metrics["collision"][3] = 0.0
        metrics["non_traversable"][3] = 0.0
        metrics["path_length"] = np.linspace(0.1, 0.9, 11, dtype=np.float32)
        sample = make_sample(
            image=np.zeros((48, 64, 3), dtype=np.uint8),
            task_text="choose a safe route",
            source_id=0,
            reference_path_ego=reference,
            metric_dict=metrics,
            goal_state=np.asarray([10, 0, 10, 0], dtype=np.float32),
            ego_state=np.zeros(8, dtype=np.float32),
            candidate_valid=np.ones(11, dtype=bool),
            sample_id="unit-test",
            lateral_span=2.5,
        )
        task = OptimizationTask(
            name="test",
            domain="nuplan",
            weights={name: 1.0 for name in METRIC_NAMES},
            limits={"collision": 0.05, "non_traversable": 0.05},
            constraint_penalty=100.0,
            explanation="unit test",
        )
        result = solve_candidate_milp(
            sample["candidate_metrics"], sample["candidate_valid"], task
        )
        self.assertEqual(result.selected_index, 3)
        self.assertEqual(result.constraint_slacks["collision"], 0.0)


if __name__ == "__main__":
    unittest.main()
