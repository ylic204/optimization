"""Portable V8.3 sample written by nuPlan or Habitat export jobs."""

from pathlib import Path

import numpy as np

from mapped_planning_schema_v82 import (
    EGO_STATE_DIM,
    GOAL_STATE_DIM,
    N_CANDIDATES,
    TRAJECTORY_STEPS,
    make_frenet_candidates,
    trajectory_geometry_features,
)
from optimization_spec_v83 import METRIC_NAMES


SCHEMA_VERSION = 83
METRIC_DIM = len(METRIC_NAMES)


def metric_matrix(metric_dict):
    """Stack named lower-is-better simulator metrics in the fixed order."""
    return np.stack(
        [np.asarray(metric_dict[name], dtype=np.float32) for name in METRIC_NAMES],
        axis=-1,
    )


def make_sample(
    image,
    task_text,
    source_id,
    reference_path_ego,
    metric_dict,
    goal_state,
    ego_state,
    candidate_valid,
    sample_id,
    lateral_span,
):
    """Create one training sample from a map route and simulator labels."""
    trajectories = make_frenet_candidates(
        reference_path_ego,
        n_candidates=N_CANDIDATES,
        steps=TRAJECTORY_STEPS,
        lateral_span=lateral_span,
    )
    return {
        "schema_version": np.int64(SCHEMA_VERSION),
        "sample_id": np.str_(sample_id),
        "task_text": np.str_(task_text),
        "source_id": np.int64(source_id),
        "image": np.asarray(image, dtype=np.uint8),
        "candidate_trajectories": trajectories,
        "candidate_features": trajectory_geometry_features(trajectories),
        "candidate_metrics": metric_matrix(metric_dict),
        "candidate_valid": np.asarray(candidate_valid, dtype=bool),
        "goal_state": np.asarray(goal_state, dtype=np.float32),
        "ego_state": np.asarray(ego_state, dtype=np.float32),
    }


def save_sample(path, sample):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **sample)
