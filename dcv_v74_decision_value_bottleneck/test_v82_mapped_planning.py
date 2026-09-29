import tempfile
import unittest
from pathlib import Path

import numpy as np

from mapped_planning_schema_v82 import (
    CANDIDATE_FEATURE_DIM,
    N_CANDIDATES,
    TRAJECTORY_STEPS,
    build_sample,
    make_frenet_candidates,
    save_sample,
    validate_sample,
)


class MappedPlanningSchemaTest(unittest.TestCase):
    def test_continuous_candidates_and_roundtrip(self):
        x = np.linspace(0.0, 20.0, 30, dtype=np.float32)
        reference = np.stack([x, 0.3 * np.sin(x / 5.0)], axis=-1)
        candidates = make_frenet_candidates(reference)
        self.assertEqual(candidates.shape, (N_CANDIDATES, TRAJECTORY_STEPS, 3))
        self.assertTrue(np.allclose(candidates[:, 0], candidates[0, 0]))
        step = np.linalg.norm(np.diff(candidates[:, :, :2], axis=1), axis=-1)
        self.assertTrue(np.all(step > 0.0))

        costs = np.linspace(1.0, 2.0, N_CANDIDATES, dtype=np.float32)
        costs[N_CANDIDATES // 2] = 0.2
        sample = build_sample(
            image=np.zeros((64, 96, 3), dtype=np.uint8),
            task_text="choose a path",
            source_id=0,
            candidate_trajectories=candidates,
            candidate_costs=costs,
            goal_state=np.asarray([20.0, 0.0, 20.0, 0.0]),
            ego_state=np.zeros(8, dtype=np.float32),
        )
        self.assertEqual(
            sample["candidate_features"].shape,
            (N_CANDIDATES, CANDIDATE_FEATURE_DIM),
        )
        validate_sample(sample)
        with tempfile.TemporaryDirectory() as directory:
            path = save_sample(Path(directory) / "sample.npz", sample)
            with np.load(path, allow_pickle=False) as loaded:
                validate_sample({key: loaded[key] for key in loaded.files})


if __name__ == "__main__":
    unittest.main()
