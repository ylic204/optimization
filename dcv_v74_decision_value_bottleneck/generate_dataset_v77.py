import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import CFG
from generate_dataset_v76 import build_one as build_v76
from graph_utils import edge_graph_features
from render_utils_v7 import render_scene


def build_one(rng, cfg):
    sample = build_v76(rng, cfg)

    # V7.6 randomly hid the edge-to-region permutation. Once graph features are
    # removed from the student, that makes the task unidentifiable. V7.7 uses a
    # fixed observable spatial layout: region position identifies the edge in
    # this controlled fixed-topology experiment.
    edge_patch = np.arange(cfg.n_edges, dtype=np.int64)
    patch_state = np.zeros(cfg.n_patches, dtype=np.int64)
    patch_state[edge_patch] = sample["edge_state"]
    distractors = np.arange(cfg.n_edges, cfg.n_patches)
    if distractors.size:
        active = rng.random(distractors.size) < 0.35
        probabilities = np.asarray(
            [
                cfg.rough_given_abnormal,
                cfg.hazard_given_abnormal,
                cfg.blocked_given_abnormal,
            ],
            dtype=np.float64,
        )
        probabilities /= probabilities.sum()
        states = rng.choice(
            np.asarray([1, 2, 3]),
            size=distractors.size,
            p=probabilities,
        )
        patch_state[distractors[active]] = states[active]

    patch_graph_feat = np.zeros((cfg.n_patches, 4), dtype=np.float32)
    edge_features = edge_graph_features(sample["base_cost"], sample["path_mask"])
    for edge, patch in enumerate(edge_patch):
        patch_graph_feat[patch, 0] = 1.0
        patch_graph_feat[patch, 1:] = edge_features[edge]

    sample["edge_patch"] = edge_patch
    sample["patch_state"] = patch_state
    sample["patch_graph_feat"] = patch_graph_feat
    sample["image"] = render_scene(patch_state, cfg, rng).astype(np.uint8)
    sample["observable_layout"] = np.asarray("fixed_edge_region_v1")
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data_v77")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    output = Path(args.out) / args.split
    output.mkdir(parents=True, exist_ok=True)
    for old_file in output.glob("*.npz"):
        old_file.unlink()
    for index in tqdm(range(args.n), desc=f"generate V7.7 {args.split}"):
        np.savez_compressed(output / f"{index:06d}.npz", **build_one(rng, CFG))
    print(f"saved {args.n} V7.7 scenes to {output}")


if __name__ == "__main__":
    main()
