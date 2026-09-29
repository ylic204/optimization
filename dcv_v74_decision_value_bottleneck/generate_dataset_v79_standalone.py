"""Standalone V7.9 generator with no dependency on older dataset versions."""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import CFG
from graph_utils import choose_path, edge_graph_features
from render_utils_v79 import render_spatial_corridor
from spatial_paths_v79 import (
    N_SPATIAL_EDGES,
    N_SPATIAL_PATHS,
    spatial_parallel_corridor,
)


TASK_PROFILES = (
    {
        "name": "balanced",
        "text": "Travel from the green S hub to the red G hub through the visible corridor. Minimize total travel cost while balancing rough terrain, hazards, and blocked edge patches.",
        "risks": (0.0, 0.35, 1.0, 20.0),
    },
    {
        "name": "safety_first",
        "text": "Travel from the green S hub to the red G hub through the visible corridor. Prioritize safety and strongly avoid rough, hazardous, or blocked edge patches.",
        "risks": (0.0, 0.80, 2.50, 25.0),
    },
    {
        "name": "efficiency_first",
        "text": "Travel from the green S hub to the red G hub through the visible corridor. Prefer the shortest efficient route, but never use a blocked edge patch.",
        "risks": (0.0, 0.12, 0.40, 20.0),
    },
)


def costs_for_task(base_cost, states, risks):
    """Compute task-conditioned edge costs without an old-version import."""
    risk = np.asarray(risks, dtype=np.float32)
    costs = np.asarray(base_cost, dtype=np.float32) + risk[states]
    traversable = (states != 3).astype(np.float32)
    return costs.astype(np.float32), traversable


def coarse_oracle_regret(
    base_cost,
    states,
    path_mask,
    true_edge_cost,
    optimal_cost,
    cfg,
):
    """Regret of a low-resolution normal/abnormal-only path oracle."""
    coarse_edge_cost = (
        np.asarray(base_cost, dtype=np.float32)
        + (states != 0).astype(np.float32) * cfg.preview_abnormal_penalty
    )
    coarse_index, _, _ = choose_path(path_mask, coarse_edge_cost)
    true_path_cost = path_mask @ true_edge_cost
    regret = max(
        (float(true_path_cost[coarse_index]) - float(optimal_cost))
        / max(float(optimal_cost), 1e-6),
        0.0,
    )
    return coarse_index, regret


def sample_edge_states(rng, cfg):
    states = np.zeros(N_SPATIAL_EDGES, dtype=np.int64)
    abnormal = rng.random(N_SPATIAL_EDGES) < cfg.edge_abnormal_prob
    probabilities = np.asarray(
        [
            cfg.rough_given_abnormal,
            cfg.hazard_given_abnormal,
            cfg.blocked_given_abnormal,
        ],
        dtype=np.float64,
    )
    probabilities /= probabilities.sum()
    sampled = rng.choice(
        np.asarray([1, 2, 3], dtype=np.int64),
        size=N_SPATIAL_EDGES,
        p=probabilities,
    )
    states[abnormal] = sampled[abnormal]
    return states


def build_one(rng, cfg):
    topology = spatial_parallel_corridor(rng, cfg.grid)
    profile = TASK_PROFILES[int(rng.integers(len(TASK_PROFILES)))]
    require_critical = rng.random() < cfg.critical_scene_fraction
    last = None
    for _ in range(cfg.max_generation_attempts):
        states = sample_edge_states(rng, cfg)
        true_edge_cost, true_traversable = costs_for_task(
            topology["base_cost"], states, profile["risks"]
        )
        optimal_index, optimal_cost, _ = choose_path(
            topology["path_mask"], true_edge_cost
        )
        coarse_index, coarse_regret = coarse_oracle_regret(
            topology["base_cost"],
            states,
            topology["path_mask"],
            true_edge_cost,
            optimal_cost,
            cfg,
        )
        last = (
            states,
            true_edge_cost,
            true_traversable,
            optimal_index,
            optimal_cost,
            coarse_index,
            coarse_regret,
        )
        if (not require_critical) or coarse_regret >= cfg.critical_min_coarse_regret:
            break

    (
        states,
        true_edge_cost,
        true_traversable,
        optimal_index,
        optimal_cost,
        coarse_index,
        coarse_regret,
    ) = last

    edge_patch = topology["edge_patch"]
    hub_patches = topology["hub_patches"]
    patch_state = np.zeros(cfg.n_patches, dtype=np.int64)
    patch_state[edge_patch] = states
    reserved = np.concatenate([edge_patch, hub_patches])
    distractors = np.setdiff1d(np.arange(cfg.n_patches), reserved)
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
        distractor_states = rng.choice(
            np.asarray([1, 2, 3]), size=distractors.size, p=probabilities
        )
        patch_state[distractors[active]] = distractor_states[active]

    patch_graph_feat = np.zeros((cfg.n_patches, 4), dtype=np.float32)
    edge_features = edge_graph_features(
        topology["base_cost"], topology["path_mask"]
    )
    for edge_index, patch in enumerate(edge_patch):
        patch_graph_feat[patch, 0] = 1.0
        patch_graph_feat[patch, 1:] = edge_features[edge_index]

    route_patch_mask = np.zeros(cfg.n_patches, dtype=np.float32)
    route_patch_mask[reserved] = 1.0
    optimal_edge_indices = topology["path_edge_indices"][optimal_index]
    optimal_edge_patches = edge_patch[optimal_edge_indices]
    optimal_path_patches = topology["path_patch_sequence"][optimal_index]
    image = render_spatial_corridor(patch_state, topology, cfg, rng)

    return {
        "image": image,
        "edges": topology["edges"],
        "base_cost": topology["base_cost"],
        "edge_patch": edge_patch,
        "edge_state": states,
        "edge_stage": topology["edge_stage"],
        "edge_choice": topology["edge_choice"],
        "patch_state": patch_state,
        "path_mask": topology["path_mask"],
        "path_edge_indices": topology["path_edge_indices"],
        "path_patch_sequence": topology["path_patch_sequence"],
        "hub_patches": hub_patches,
        "route_patch_mask": route_patch_mask,
        "true_edge_cost": true_edge_cost,
        "true_traversable": true_traversable,
        "optimal_path_idx": np.int64(optimal_index),
        "optimal_cost": np.float32(optimal_cost),
        "optimal_edge_mask": topology["path_mask"][optimal_index],
        "optimal_edge_patches": optimal_edge_patches,
        "optimal_path_patches": optimal_path_patches,
        # Retained only for old teacher/evaluation compatibility.  The Qwen
        # selector never receives this privileged tensor.
        "patch_graph_feat": patch_graph_feat,
        "coarse_oracle_path_idx": np.int64(coarse_index),
        "coarse_oracle_regret": np.float32(coarse_regret),
        "critical_scene": np.int64(
            coarse_regret >= cfg.critical_min_coarse_regret
        ),
        "task_name": np.asarray(profile["name"]),
        "task_text": np.asarray(profile["text"]),
        "task_risks": np.asarray(profile["risks"], dtype=np.float32),
        "observable_layout": np.asarray("spatial_parallel_corridor_v1"),
        "spatial_grid": np.int64(cfg.grid),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data_v79")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if CFG.grid < 9:
        raise ValueError("Set GRID=9 (or larger) before generating V7.9 data")
    if CFG.n_paths != N_SPATIAL_PATHS:
        raise ValueError(f"V7.9 requires n_paths={N_SPATIAL_PATHS}")

    rng = np.random.default_rng(args.seed)
    output = Path(args.out) / args.split
    output.mkdir(parents=True, exist_ok=True)
    for old_file in output.glob("*.npz"):
        old_file.unlink()
    for index in tqdm(range(args.n), desc=f"generate V7.9 {args.split}"):
        np.savez_compressed(output / f"{index:06d}.npz", **build_one(rng, CFG))
    print(
        f"saved {args.n} V7.9 scenes to {output}; grid={CFG.grid}x{CFG.grid}, "
        f"edges={N_SPATIAL_EDGES}, continuous_paths={N_SPATIAL_PATHS}"
    )


if __name__ == "__main__":
    main()
