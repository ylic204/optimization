"""Shared sample schema for nuPlan and Habitat PointNav planning.

The online planner sees RGB, task text, ego/goal state and candidate
trajectories.  ``candidate_costs`` are downstream-task supervision and must
never be passed to the student model.
"""

from pathlib import Path

import numpy as np


SCHEMA_VERSION = 82
N_CANDIDATES = 11
TRAJECTORY_STEPS = 16
CANDIDATE_FEATURE_DIM = 8
GOAL_STATE_DIM = 4
EGO_STATE_DIM = 8

SOURCE_NUPLAN = 0
SOURCE_POINTNAV = 1


def wrap_angle(angle):
    """Wrap radians to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def _resample_polyline(reference, steps):
    reference = np.asarray(reference, dtype=np.float32)
    if reference.ndim != 2 or reference.shape[1] not in (2, 3):
        raise ValueError("reference must have shape [N,2] or [N,3]")
    if len(reference) < 2:
        raise ValueError("reference needs at least two points")

    xy = reference[:, :2]
    segment = np.linalg.norm(np.diff(xy, axis=0), axis=-1)
    arc = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(segment)])
    if arc[-1] <= 1e-6:
        raise ValueError("reference path has zero length")
    target = np.linspace(0.0, float(arc[-1]), int(steps), dtype=np.float32)
    x = np.interp(target, arc, xy[:, 0])
    y = np.interp(target, arc, xy[:, 1])
    if reference.shape[1] == 3:
        heading = np.unwrap(reference[:, 2])
        yaw = np.interp(target, arc, heading)
    else:
        dx = np.gradient(x)
        dy = np.gradient(y)
        yaw = np.arctan2(dy, dx)
    return np.stack([x, y, wrap_angle(yaw)], axis=-1).astype(np.float32)


def make_frenet_candidates(
    reference,
    n_candidates=N_CANDIDATES,
    steps=TRAJECTORY_STEPS,
    lateral_span=2.5,
):
    """Create spatially continuous ego-frame candidates around a route.

    The middle trajectory follows ``reference``.  The others gradually fan
    out to evenly spaced lateral offsets; all candidates start from the same
    ego pose.  Coordinates follow x-forward, y-left, heading-radians.
    """
    if int(n_candidates) < 1 or int(n_candidates) % 2 != 1:
        raise ValueError("n_candidates must be a positive odd number")
    center = _resample_polyline(reference, steps)
    progress = np.linspace(0.0, 1.0, int(steps), dtype=np.float32)
    blend = progress * progress * (3.0 - 2.0 * progress)
    offsets = np.linspace(
        -float(lateral_span),
        float(lateral_span),
        int(n_candidates),
        dtype=np.float32,
    )
    candidates = []
    for offset in offsets:
        lateral = offset * blend
        x = center[:, 0] - np.sin(center[:, 2]) * lateral
        y = center[:, 1] + np.cos(center[:, 2]) * lateral
        yaw = np.arctan2(np.gradient(y), np.gradient(x))
        trajectory = np.stack([x, y, wrap_angle(yaw)], axis=-1)
        trajectory[0] = center[0]
        candidates.append(trajectory.astype(np.float32))
    return np.stack(candidates, axis=0)


def trajectory_geometry_features(candidate_trajectories):
    """Return deployable geometry features, never teacher-only costs.

    Feature order: length, end-x, end-y, sin(end-yaw), cos(end-yaw),
    mean absolute curvature, max absolute curvature, terminal lateral offset.
    """
    trajectories = np.asarray(candidate_trajectories, dtype=np.float32)
    if trajectories.ndim != 3 or trajectories.shape[-1] != 3:
        raise ValueError("candidate_trajectories must have shape [P,H,3]")
    delta = np.diff(trajectories[:, :, :2], axis=1)
    ds = np.linalg.norm(delta, axis=-1).clip(min=1e-4)
    length = ds.sum(axis=-1)
    d_yaw = wrap_angle(np.diff(trajectories[:, :, 2], axis=1))
    curvature = np.abs(d_yaw) / ds
    end = trajectories[:, -1]
    return np.stack(
        [
            length,
            end[:, 0],
            end[:, 1],
            np.sin(end[:, 2]),
            np.cos(end[:, 2]),
            curvature.mean(axis=-1),
            curvature.max(axis=-1),
            end[:, 1],
        ],
        axis=-1,
    ).astype(np.float32)


def validate_sample(sample):
    """Validate a standardized sample and return it unchanged."""
    required = {
        "task_text",
        "source_id",
        "candidate_trajectories",
        "candidate_features",
        "candidate_valid",
        "candidate_costs",
        "optimal_path_idx",
        "goal_state",
        "ego_state",
    }
    missing = sorted(required - set(sample))
    if missing:
        raise KeyError(f"missing V8.2 fields: {missing}")
    if "image" not in sample and "image_path" not in sample:
        raise KeyError("sample needs image or image_path")

    trajectories = np.asarray(sample["candidate_trajectories"])
    if trajectories.shape != (N_CANDIDATES, TRAJECTORY_STEPS, 3):
        raise ValueError(
            "candidate_trajectories must have shape "
            f"{(N_CANDIDATES, TRAJECTORY_STEPS, 3)}, got {trajectories.shape}"
        )
    features = np.asarray(sample["candidate_features"])
    if features.shape != (N_CANDIDATES, CANDIDATE_FEATURE_DIM):
        raise ValueError("candidate_features has the wrong shape")
    valid = np.asarray(sample["candidate_valid"], dtype=bool)
    costs = np.asarray(sample["candidate_costs"], dtype=np.float32)
    if valid.shape != (N_CANDIDATES,) or costs.shape != (N_CANDIDATES,):
        raise ValueError("candidate_valid/candidate_costs must have shape [11]")
    if not valid.any() or not np.isfinite(costs[valid]).all():
        raise ValueError("every sample needs at least one finite valid cost")
    optimal = int(np.asarray(sample["optimal_path_idx"]).item())
    if not 0 <= optimal < N_CANDIDATES or not valid[optimal]:
        raise ValueError("optimal_path_idx must identify a valid candidate")
    if int(np.argmin(np.where(valid, costs, np.inf))) != optimal:
        raise ValueError("optimal_path_idx does not match candidate_costs")
    if np.asarray(sample["goal_state"]).shape != (GOAL_STATE_DIM,):
        raise ValueError("goal_state must be [dx,dy,distance,bearing]")
    if np.asarray(sample["ego_state"]).shape != (EGO_STATE_DIM,):
        raise ValueError("ego_state must have 8 values")
    return sample


def build_sample(
    image,
    task_text,
    source_id,
    candidate_trajectories,
    candidate_costs,
    goal_state,
    ego_state,
    candidate_valid=None,
    sample_id="",
    image_path=None,
):
    """Construct a serializable V8.2 sample dictionary."""
    trajectories = np.asarray(candidate_trajectories, dtype=np.float32)
    valid = (
        np.ones(N_CANDIDATES, dtype=bool)
        if candidate_valid is None
        else np.asarray(candidate_valid, dtype=bool)
    )
    costs = np.asarray(candidate_costs, dtype=np.float32)
    optimal = int(np.argmin(np.where(valid, costs, np.inf)))
    sample = {
        "schema_version": np.int64(SCHEMA_VERSION),
        "task_text": np.str_(task_text),
        "source_id": np.int64(source_id),
        "sample_id": np.str_(sample_id),
        "candidate_trajectories": trajectories,
        "candidate_features": trajectory_geometry_features(trajectories),
        "candidate_valid": valid,
        "candidate_costs": costs,
        "optimal_path_idx": np.int64(optimal),
        "goal_state": np.asarray(goal_state, dtype=np.float32),
        "ego_state": np.asarray(ego_state, dtype=np.float32),
    }
    if image is not None:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("image must be RGB HWC")
        sample["image"] = image.astype(np.uint8)
    elif image_path is not None:
        sample["image_path"] = np.str_(image_path)
    validate_sample(sample)
    return sample


def save_sample(path, sample):
    """Validate and atomically save one compressed NPZ sample."""
    validate_sample(sample)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **sample)
    temporary.replace(path)
    return path
