import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from planning_data_v84 import SDF_SIZE, convert_directory, process_raw_record
from raw_record_v84 import save_raw_record
from split_planning_data_v84 import split_groups


class PlanningDataV84Test(unittest.TestCase):
    @staticmethod
    def make_raw():
        traversable = np.ones((64, 64), dtype=bool)
        obstacle = np.zeros((64, 64), dtype=bool)
        obstacle[28:36, 40:44] = True
        reference = np.stack(
            [np.linspace(0, 15, 24), np.zeros(24)], axis=-1
        ).astype(np.float32)
        return {
            "image": np.zeros((72, 96, 3), dtype=np.uint8),
            "reference_path_ego": reference,
            "traversable": traversable,
            "dynamic_obstacle": obstacle,
            "map_bounds": np.asarray([-5, 20, -10, 10], dtype=np.float32),
            "goal_xy": np.asarray([15, 0], dtype=np.float32),
            "ego_state": np.zeros(8, dtype=np.float32),
            "task_text": np.asarray("find a safe path"),
            "source_id": np.asarray(0),
            "sample_id": np.asarray("sample-0"),
        }

    def test_raw_record_to_training_sample(self):
        raw = self.make_raw()
        sample = process_raw_record(raw, lateral_span=2.5)
        self.assertEqual(sample["candidate_trajectories"].shape, (11, 16, 3))
        self.assertEqual(sample["candidate_metrics"].shape, (11, 8))
        self.assertTrue(sample["candidate_valid"].any())
        self.assertEqual(sample["sdf"].shape, (SDF_SIZE, SDF_SIZE))
        self.assertLess(sample["sdf"].min(), 0.0)
        self.assertGreater(sample["sdf"].max(), 0.0)

    def test_save_and_convert_directory(self):
        raw = self.make_raw()
        with TemporaryDirectory() as directory:
            raw_root = Path(directory) / "raw"
            output_root = Path(directory) / "processed"
            save_raw_record(
                raw_root / "scene" / "sample.npz",
                image=raw["image"],
                reference_path_ego=raw["reference_path_ego"],
                traversable=raw["traversable"],
                dynamic_obstacle=raw["dynamic_obstacle"],
                map_bounds=raw["map_bounds"],
                goal_xy=raw["goal_xy"],
                ego_state=raw["ego_state"],
                task_text=raw["task_text"].item(),
                source_id=raw["source_id"].item(),
                sample_id=raw["sample_id"].item(),
            )
            convert_directory(raw_root, output_root, "nuplan")
            with np.load(output_root / "scene" / "sample.npz") as sample:
                self.assertEqual(int(sample["schema_version"]), 84)
                self.assertEqual(sample["sample_id"].item(), "sample-0")
                self.assertEqual(sample["candidate_metrics"].shape, (11, 8))

    def test_group_split_keeps_scenes_together(self):
        raw = self.make_raw()
        with TemporaryDirectory() as directory:
            input_root = Path(directory) / "all"
            output_root = Path(directory) / "split"
            for scene in range(6):
                for frame in range(2):
                    save_raw_record(
                        input_root / f"scene-{scene}" / f"{frame}.npz",
                        image=raw["image"],
                        reference_path_ego=raw["reference_path_ego"],
                        traversable=raw["traversable"],
                        dynamic_obstacle=raw["dynamic_obstacle"],
                        map_bounds=raw["map_bounds"],
                        goal_xy=raw["goal_xy"],
                        ego_state=raw["ego_state"],
                        task_text=raw["task_text"].item(),
                        source_id=0,
                        sample_id=f"scene-{scene}/{frame}",
                    )
            manifest = split_groups(
                input_root, output_root, 0.2, 0.2, 2026, 1
            )
            self.assertEqual(sum(manifest["sample_counts"].values()), 12)
            for scene in range(6):
                locations = [
                    split
                    for split in ("train", "val", "test")
                    if (output_root / split / f"scene-{scene}").exists()
                ]
                self.assertEqual(len(locations), 1)


if __name__ == "__main__":
    unittest.main()
