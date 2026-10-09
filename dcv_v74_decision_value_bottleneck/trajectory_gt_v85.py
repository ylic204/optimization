"""Time-aligned rear-axle targets; future labels are never planner inputs."""

import numpy as np


GT_CONTRACT = "ego_rear_axle_future_seconds_v1"


def aligned_expert(raw, steps=16):
    """Interpolate recorded poses in time, including t=0; never extrapolate."""
    names = ("expert_trajectory_ego", "expert_time_s", "planning_time_s")
    if not all(name in raw for name in names):
        return np.zeros((steps, 3), np.float32), np.zeros(steps, np.float32), False
    poses = np.asarray(raw[names[0]], dtype=np.float64)
    times = np.asarray(raw[names[1]], dtype=np.float64)
    query = np.asarray(raw[names[2]], dtype=np.float64)
    if poses.shape != (len(times), 3) or query.shape != (steps,):
        raise ValueError("GT poses/timestamps or planning grid have invalid shapes")
    if not all(np.isfinite(x).all() for x in (poses, times, query)):
        raise ValueError("GT must contain finite poses and timestamps")
    if len(times) < 2 or (times <= 0).any() or (np.diff(times) <= 0).any():
        raise ValueError("expert_time_s must strictly increase after the current frame")
    if (query <= 0).any() or (np.diff(query) <= 0).any() or query[-1] > times[-1] + 1e-7:
        raise ValueError("planning times must increase and be covered by recorded future GT")
    times = np.r_[0.0, times]
    poses = np.vstack([np.zeros((1, 3)), poses])
    poses[:, 2] = np.unwrap(poses[:, 2])
    aligned = np.stack([np.interp(query, times, poses[:, i]) for i in range(3)], -1)
    aligned[:, 2] = (aligned[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return aligned.astype(np.float32), query.astype(np.float32), True


def displacement_errors(trajectory, expert):
    trajectory, expert = np.asarray(trajectory), np.asarray(expert)
    if trajectory.shape != expert.shape or trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("trajectory and GT must share [N,2] at the same timestamps")
    if not np.isfinite(trajectory).all() or not np.isfinite(expert).all():
        raise ValueError("trajectory and GT must be finite")
    errors = np.linalg.norm(trajectory - expert, axis=-1)
    return float(errors.mean()), float(errors[-1])
