from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class DCVDataset(Dataset):
    def __init__(self, root, teacher_root=None):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.npz"))
        self.teacher_root = Path(teacher_root) if teacher_root else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        d = np.load(p)
        out = {
            "name": p.stem,
            "image": torch.from_numpy(d["image"]).permute(2,0,1).float()/255.0,
            "edges": torch.from_numpy(d["edges"]).long(),
            "base_cost": torch.from_numpy(d["base_cost"]).float(),
            "edge_patch": torch.from_numpy(d["edge_patch"]).long(),
            "edge_state": torch.from_numpy(d["edge_state"]).long(),
            "patch_state": torch.from_numpy(d["patch_state"]).long(),
            "path_mask": torch.from_numpy(d["path_mask"]).float(),
            "true_edge_cost": torch.from_numpy(d["true_edge_cost"]).float(),
            "optimal_path_idx": torch.tensor(int(d["optimal_path_idx"]), dtype=torch.long),
            "optimal_cost": torch.tensor(float(d["optimal_cost"]), dtype=torch.float32),
            "patch_graph_feat": torch.from_numpy(d["patch_graph_feat"]).float(),
        }
        # V7.6 task conditioning. Older datasets remain readable and receive
        # the original balanced navigation objective as their default prompt.
        out["task_text"] = str(d["task_text"].item()) if "task_text" in d.files else (
            "Navigate from node 0 to node 13. Minimize travel cost while "
            "balancing rough terrain, hazards, and blocked regions."
        )
        out["task_name"] = str(d["task_name"].item()) if "task_name" in d.files else "balanced"
        out["task_risks"] = torch.from_numpy(d["task_risks"]).float() if "task_risks" in d.files else torch.tensor(
            [0.0, 0.35, 1.0, 20.0], dtype=torch.float32
        )
        if "coarse_oracle_regret" in d.files:
            out["coarse_oracle_regret"] = torch.tensor(float(d["coarse_oracle_regret"]), dtype=torch.float32)
        # Optional V7.9 spatial-topology fields.  They are metadata and
        # supervision, not inputs to the student selector.
        for key in (
            "edge_stage",
            "edge_choice",
            "path_edge_indices",
            "path_patch_sequence",
            "hub_patches",
            "optimal_edge_patches",
            "optimal_path_patches",
        ):
            if key in d.files:
                out[key] = torch.from_numpy(d[key]).long()
        for key in ("route_patch_mask", "optimal_edge_mask"):
            if key in d.files:
                out[key] = torch.from_numpy(d[key]).float()
        if "observable_layout" in d.files:
            out["observable_layout"] = str(d["observable_layout"].item())
        if "spatial_grid" in d.files:
            out["spatial_grid"] = torch.tensor(int(d["spatial_grid"]), dtype=torch.long)
        if self.teacher_root is not None:
            td = np.load(self.teacher_root/f"{p.stem}.npz")
            for key in td.files:
                arr=td[key]
                out[key]=torch.from_numpy(arr).long() if arr.dtype.kind in ("i","u") else torch.from_numpy(arr).float()
        return out
