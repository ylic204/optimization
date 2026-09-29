"""PyTorch dataset consumed by the complete V8.4 training pipeline."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PlanningDatasetV84(Dataset):
    def __init__(self, root, image_side=288):
        self.files = sorted(Path(root).rglob("*.npz"))
        self.image_side = image_side

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index], allow_pickle=False) as data:
            image = Image.fromarray(data["image"]).convert("RGB")
            image = image.resize(
                (self.image_side, self.image_side), Image.Resampling.BICUBIC
            )
            image = np.asarray(image, dtype=np.uint8).copy()
            return {
                "image": torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,
                "task_text": str(data["task_text"].item()),
                "source_id": torch.tensor(data["source_id"].item()).long(),
                "candidate_trajectories": torch.from_numpy(
                    data["candidate_trajectories"].astype(np.float32)
                ),
                "candidate_features": torch.from_numpy(
                    data["candidate_features"].astype(np.float32)
                ),
                "candidate_metrics": torch.from_numpy(
                    data["candidate_metrics"].astype(np.float32)
                ),
                "candidate_valid": torch.from_numpy(
                    data["candidate_valid"].astype(bool)
                ),
                "candidate_count": torch.tensor(
                    data["candidate_count"].item()
                ).long(),
                "expert_trajectory": torch.from_numpy(
                    data["expert_trajectory"].astype(np.float32)
                ),
                "expert_is_fallback": torch.tensor(
                    data["expert_is_fallback"].item()
                ).bool(),
                "goal_state": torch.from_numpy(
                    data["goal_state"].astype(np.float32)
                ),
                "ego_state": torch.from_numpy(
                    data["ego_state"].astype(np.float32)
                ),
                "sdf": torch.from_numpy(data["sdf"].astype(np.float32))[None],
                "map_bounds": torch.from_numpy(
                    data["map_bounds"].astype(np.float32)
                ),
                "sample_id": str(data["sample_id"].item()),
            }
