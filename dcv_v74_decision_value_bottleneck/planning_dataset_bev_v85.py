"""BEV-only PyTorch dataset for V8.5 gradient-flow token selection."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from raw_record_bev_v85 import INPUT_MODE


class PlanningDatasetBEVV85(Dataset):
    def __init__(self, root, image_side=288):
        self.files = sorted(Path(root).rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no processed BEV NPZ files under {root}")
        self.image_side = int(image_side)
        with np.load(self.files[0], allow_pickle=False) as data:
            self.bev_config_json = self._validate_metadata(data, self.files[0])
            self.bev_config = json.loads(self.bev_config_json)

    def _validate_metadata(self, data, path):
        input_mode = str(data["input_mode"].item()) if "input_mode" in data else ""
        if input_mode != INPUT_MODE:
            raise ValueError(
                f"{path} is {input_mode or 'legacy RGB'}, not {INPUT_MODE}"
            )
        if "bev_config_json" not in data:
            raise KeyError(f"{path} does not contain bev_config_json")
        value = str(data["bev_config_json"].item())
        config = json.loads(value)
        if int(config["image_size"]) != self.image_side:
            raise ValueError(
                f"{path} uses image_size={config['image_size']}, expected "
                f"{self.image_side}; BEV samples must not be resized"
            )
        expected_grid = self.image_side // 32
        if int(config["region_grid"]) != expected_grid:
            raise ValueError(
                f"{path} uses region_grid={config['region_grid']}, expected "
                f"{expected_grid} for 32-pixel selector regions"
            )
        return value

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index], allow_pickle=False) as data:
            config_json = self._validate_metadata(data, self.files[index])
            if config_json != self.bev_config_json:
                raise ValueError(
                    f"{self.files[index]} uses a different BEV geometry; "
                    "one dataset split must use one fixed metric frame"
                )
            image = np.asarray(data["bev_rgb"], dtype=np.uint8)
            if image.shape != (self.image_side, self.image_side, 3):
                raise ValueError(
                    f"BEV image {image.shape} does not match the required "
                    f"{self.image_side}x{self.image_side}; fixed metric scale forbids resizing"
                )
            region_bounds = data["region_world_bounds"].astype(np.float32)
            expected_regions = self.bev_config["region_grid"] ** 2
            if region_bounds.shape != (expected_regions, 4):
                raise ValueError(
                    f"{self.files[index]} has invalid region_world_bounds "
                    f"shape {region_bounds.shape}"
                )
            image = image.copy()
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
                "region_world_bounds": torch.from_numpy(
                    region_bounds
                ),
                "sample_id": str(data["sample_id"].item()),
                "scene_type": str(data["scene_type"].item()),
            }
