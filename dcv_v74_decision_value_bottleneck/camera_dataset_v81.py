"""PyTorch dataset for V8.1 first-person local-trajectory samples."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _rgb_tensor(array):
    return torch.from_numpy(np.asarray(array).copy()).permute(2, 0, 1).float() / 255.0


class CameraTrajectoryDataset(Dataset):
    """Load RGB as the student input and keep geometry as supervision.

    ``image`` is always first-person RGB and is the only tensor intended for
    the default Qwen3-VL visual stream.  Depth/segmentation are returned as GT
    tensors.  Their colorized RGB versions are loaded only when explicitly
    requested for a later multi-image ablation.
    """

    def __init__(
        self,
        root,
        task_text=None,
        include_dense_mapping=False,
        include_vlm_aux_images=False,
    ):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.npz"))
        self.task_text = task_text
        self.include_dense_mapping = bool(include_dense_mapping)
        self.include_vlm_aux_images = bool(include_vlm_aux_images)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with np.load(path) as data:
            rgb = _rgb_tensor(data["rgb"])
            out = {
                "name": path.stem,
                "image": rgb,
                "rgb": rgb,
                "depth_gt": torch.from_numpy(data["depth_gt"].copy()).float().unsqueeze(0),
                "semantic_gt": torch.from_numpy(data["semantic_gt"].copy()).long(),
                "local_bev_gt": torch.from_numpy(data["local_bev_gt"].copy()).long(),
                "trajectory_points_robot": torch.from_numpy(
                    data["trajectory_points_robot"].copy()
                ).float(),
                "trajectory_cells": torch.from_numpy(data["trajectory_cells"].copy()).long(),
                "trajectory_lateral_offsets": torch.from_numpy(
                    data["trajectory_lateral_offsets"].copy()
                ).float(),
                "trajectory_features": torch.from_numpy(
                    data["trajectory_features"].copy()
                ).float(),
                "trajectory_region_features": torch.from_numpy(
                    data["trajectory_region_features"].copy()
                ).float(),
                "trajectory_unmapped_features": torch.from_numpy(
                    data["trajectory_unmapped_features"].copy()
                ).float(),
                "candidate_path_patch_relation": torch.from_numpy(
                    data["candidate_path_patch_relation"].copy()
                ).float(),
                "candidate_path_patch_mask": torch.from_numpy(
                    data["candidate_path_patch_mask"].copy()
                ).float(),
                "trajectory_costs": torch.from_numpy(data["trajectory_costs"].copy()).float(),
                "optimal_path_idx": torch.tensor(
                    int(data["optimal_path_idx"]), dtype=torch.long
                ),
                "optimal_cost": torch.tensor(
                    float(data["optimal_cost"]), dtype=torch.float32
                ),
                "task_risks": torch.from_numpy(data["task_risks"].copy()).float(),
                "patch_to_bev": torch.from_numpy(data["patch_to_bev"].copy()).long(),
                "patch_to_bev_valid": torch.from_numpy(
                    data["patch_to_bev_valid"].copy()
                ).float(),
                "task_text": self.task_text
                if self.task_text is not None
                else str(data["task_text"].item()),
                "student_input_mode": str(data["student_input_mode"].item()),
                "layout": str(data["layout"].item()),
            }
            if self.include_dense_mapping:
                out["pixel_to_bev_row"] = torch.from_numpy(
                    data["pixel_to_bev_row"].copy()
                ).long()
                out["pixel_to_bev_col"] = torch.from_numpy(
                    data["pixel_to_bev_col"].copy()
                ).long()
            if self.include_vlm_aux_images:
                out["depth_vlm_image"] = _rgb_tensor(data["depth_vlm_image"])
                out["semantic_vlm_image"] = _rgb_tensor(
                    data["semantic_vlm_image"]
                )
        return out
