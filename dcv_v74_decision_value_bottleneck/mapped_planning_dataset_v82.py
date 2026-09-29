"""PyTorch dataset for the unified nuPlan/PointNav V8.2 samples."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mapped_planning_schema_v82 import (
    CANDIDATE_FEATURE_DIM,
    EGO_STATE_DIM,
    GOAL_STATE_DIM,
    N_CANDIDATES,
    TRAJECTORY_STEPS,
    validate_sample,
)


class MappedPlanningDataset(Dataset):
    def __init__(self, root, task_text_override=None, image_side=288):
        self.root = Path(root)
        self.files = sorted(self.root.rglob("*.npz"))
        if not self.files:
            raise ValueError(f"no .npz samples found under {self.root}")
        self.task_text_override = task_text_override
        self.image_side = int(image_side)

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _scalar(data, key):
        return data[key].item() if np.asarray(data[key]).ndim == 0 else data[key]

    def _load_image(self, data, sample_file):
        if "image" in data:
            image = np.asarray(data["image"], dtype=np.uint8)
        else:
            image_path = Path(str(self._scalar(data, "image_path")))
            if not image_path.is_absolute():
                image_path = sample_file.parent / image_path
            with Image.open(image_path) as handle:
                image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{sample_file}: expected RGB HWC image")
        if image.shape[:2] != (self.image_side, self.image_side):
            image = np.asarray(
                Image.fromarray(image).resize(
                    (self.image_side, self.image_side), Image.Resampling.BICUBIC
                ),
                dtype=np.uint8,
            )
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, index):
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            raw = {key: data[key] for key in data.files}
            validate_sample(raw)
            text = str(self._scalar(data, "task_text"))
            if self.task_text_override:
                text = self.task_text_override
            return {
                "image": self._load_image(data, path),
                "task_text": text,
                "source_id": torch.tensor(
                    int(self._scalar(data, "source_id")), dtype=torch.long
                ),
                "candidate_trajectories": torch.from_numpy(
                    np.asarray(data["candidate_trajectories"], dtype=np.float32)
                ),
                "candidate_features": torch.from_numpy(
                    np.asarray(data["candidate_features"], dtype=np.float32)
                ),
                "candidate_valid": torch.from_numpy(
                    np.asarray(data["candidate_valid"], dtype=bool)
                ),
                "candidate_costs": torch.from_numpy(
                    np.asarray(data["candidate_costs"], dtype=np.float32)
                ),
                "optimal_path_idx": torch.tensor(
                    int(self._scalar(data, "optimal_path_idx")), dtype=torch.long
                ),
                "goal_state": torch.from_numpy(
                    np.asarray(data["goal_state"], dtype=np.float32)
                ),
                "ego_state": torch.from_numpy(
                    np.asarray(data["ego_state"], dtype=np.float32)
                ),
                "sample_id": str(self._scalar(data, "sample_id")),
                "sample_path": str(path),
            }


def assert_compatible_sample(sample):
    """Fail early before loading the 8B model."""
    expected = {
        "candidate_trajectories": (N_CANDIDATES, TRAJECTORY_STEPS, 3),
        "candidate_features": (N_CANDIDATES, CANDIDATE_FEATURE_DIM),
        "goal_state": (GOAL_STATE_DIM,),
        "ego_state": (EGO_STATE_DIM,),
    }
    for key, shape in expected.items():
        if tuple(sample[key].shape) != shape:
            raise ValueError(f"{key}: expected {shape}, got {tuple(sample[key].shape)}")
