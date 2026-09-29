import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from config import CFG
from generate_dataset_v7 import coarse_oracle_regret, sample_states
from graph_utils import choose_path, edge_graph_features, fixed_layered_graph
from render_utils_v7 import render_scene


TASK_PROFILES = (
    {
        "name": "balanced",
        "text": "Navigate from node 0 to node 13. Minimize travel cost while balancing rough terrain, hazards, and blocked regions.",
        "risks": (0.0, 0.35, 1.0, 20.0),
    },
    {
        "name": "safety_first",
        "text": "Navigate from node 0 to node 13. Prioritize safety: strongly avoid rough terrain, hazards, and blocked regions.",
        "risks": (0.0, 0.80, 2.50, 25.0),
    },
    {
        "name": "efficiency_first",
        "text": "Navigate from node 0 to node 13. Prioritize a short efficient route, but never enter blocked regions.",
        "risks": (0.0, 0.12, 0.40, 20.0),
    },
)


def costs_for_task(base_cost, states, risks):
    risk = np.asarray(risks, dtype=np.float32)
    cost = base_cost + risk[states]
    return cost.astype(np.float32), (states != 3).astype(np.float32)


def build_one(rng, cfg):
    profile = TASK_PROFILES[int(rng.integers(len(TASK_PROFILES)))]
    edges, base_cost, path_mask = fixed_layered_graph(rng)
    require_critical = rng.random() < cfg.critical_scene_fraction
    last = None
    for _ in range(cfg.max_generation_attempts):
        states = sample_states(rng, cfg)
        true_edge_cost, true_traversable = costs_for_task(
            base_cost, states, profile["risks"]
        )
        opt_idx, opt_cost, _ = choose_path(path_mask, true_edge_cost)
        coarse_idx, coarse_regret = coarse_oracle_regret(
            base_cost,
            states,
            path_mask,
            true_edge_cost,
            opt_cost,
            cfg,
        )
        last = (
            states,
            true_edge_cost,
            true_traversable,
            opt_idx,
            opt_cost,
            coarse_idx,
            coarse_regret,
        )
        if (not require_critical) or coarse_regret >= cfg.critical_min_coarse_regret:
            break

    (
        states,
        true_edge_cost,
        true_traversable,
        opt_idx,
        opt_cost,
        coarse_idx,
        coarse_regret,
    ) = last
    edge_patch = rng.choice(
        cfg.n_patches, size=cfg.n_edges, replace=False
    ).astype(np.int64)
    patch_state = np.zeros(cfg.n_patches, dtype=np.int64)
    patch_state[edge_patch] = states
    distractors = np.setdiff1d(np.arange(cfg.n_patches), edge_patch)
    if len(distractors):
        active = rng.random(len(distractors)) < 0.35
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
            np.asarray([1, 2, 3]), size=len(distractors), p=probabilities
        )
        patch_state[distractors[active]] = distractor_states[active]

    image = render_scene(patch_state, cfg, rng)
    # Kept only for teacher/evaluation compatibility. V7.6 student never reads it.
    edge_features = edge_graph_features(base_cost, path_mask)
    patch_graph_feat = np.zeros((cfg.n_patches, 4), dtype=np.float32)
    for edge, patch in enumerate(edge_patch):
        patch_graph_feat[patch, 0] = 1.0
        patch_graph_feat[patch, 1:] = edge_features[edge]

    return dict(
        image=image.astype(np.uint8),
        edges=edges.astype(np.int64),
        base_cost=base_cost.astype(np.float32),
        edge_patch=edge_patch,
        edge_state=states.astype(np.int64),
        patch_state=patch_state,
        path_mask=path_mask.astype(np.float32),
        true_edge_cost=true_edge_cost.astype(np.float32),
        true_traversable=true_traversable,
        optimal_path_idx=np.int64(opt_idx),
        optimal_cost=np.float32(opt_cost),
        optimal_edge_mask=path_mask[opt_idx].astype(np.float32),
        patch_graph_feat=patch_graph_feat,
        coarse_oracle_path_idx=np.int64(coarse_idx),
        coarse_oracle_regret=np.float32(coarse_regret),
        critical_scene=np.int64(coarse_regret >= cfg.critical_min_coarse_regret),
        task_name=np.asarray(profile["name"]),
        task_text=np.asarray(profile["text"]),
        task_risks=np.asarray(profile["risks"], dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data_v76")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    output = Path(args.out) / args.split
    output.mkdir(parents=True, exist_ok=True)
    for old_file in output.glob("*.npz"):
        old_file.unlink()
    for index in tqdm(range(args.n), desc=f"generate V7.6 {args.split}"):
        np.savez_compressed(output / f"{index:06d}.npz", **build_one(rng, CFG))
    print(f"saved {args.n} V7.6 scenes to {output}")


if __name__ == "__main__":
    main()
