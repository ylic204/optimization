"""Logged future trajectory -> a visible boundary goal and timestamped prefix.

No route completion, extrapolation or synthetic trajectory is used. Only the
single boundary intersection is interpolated between recorded future samples.
"""
import math
import numpy as np
from bev_renderer_v85 import world_to_ego, wrap_angle


def load_logged_future(scenario, iteration, ego_pose, horizon_s, interval_s):
    count = max(2, int(math.ceil(horizon_s / interval_s)))
    states = list(scenario.get_ego_future_trajectory(
        iteration, time_horizon=horizon_s, num_samples=count))
    anchor = scenario.get_time_point(iteration).time_us
    states = [s for s in states if s.time_point.time_us > anchor]
    if not states:
        raise ValueError("no_logged_future")
    times = np.r_[0., [(s.time_point.time_us-anchor)*1e-6 for s in states]]
    xy = world_to_ego([[s.rear_axle.x, s.rear_axle.y] for s in states], ego_pose)
    yaw = [float(wrap_angle(s.rear_axle.heading-ego_pose[2])) for s in states]
    poses = np.vstack([np.zeros(3), np.c_[xy, yaw]])
    if not np.isfinite(poses).all() or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("invalid_logged_future_timestamps_or_poses")
    return poses, times


def boundary_inset(vehicle, safety_margin_m, requested_margin_m):
    """Conservative rear-axle inset that contains the body at any heading."""
    longitudinal = vehicle["length"]/2 + abs(vehicle["rear_axle_to_center"]) + safety_margin_m
    lateral = vehicle["width"]/2 + safety_margin_m
    return max(requested_margin_m, math.hypot(longitudinal, lateral) + .05)


def truncate_at_boundary(poses, times, bounds, inset_m):
    """First exit from an inset rectangle; preserve stops and original times.

    Returns None if the available GT never reaches the boundary. A later re-entry
    cannot extend the prefix beyond the first exit. Output starts at t=0.
    """
    poses, times = np.asarray(poses, dtype=float), np.asarray(times, dtype=float)
    if (poses.ndim != 2 or poses.shape[1] != 3 or times.shape != (len(poses),)
            or len(poses) < 2 or not np.isfinite(poses).all() or not np.isfinite(times).all()
            or (np.diff(times) <= 0).any() or abs(times[0]) > 1e-8
            or np.linalg.norm(poses[0]) > 1e-6):
        raise ValueError("GT must be finite [N,3], start at (0,0,0), and have increasing times from zero")
    xmin, xmax, ymin, ymax = np.asarray(bounds, dtype=float)
    lower = np.array([xmin+inset_m, ymin+inset_m])
    upper = np.array([xmax-inset_m, ymax-inset_m])
    if (not np.isfinite(inset_m) or inset_m <= 0 or not np.isfinite(bounds).all()
            or not ((lower < 0).all() and (upper > 0).all())):
        raise ValueError("GT crop must contain the complete start footprint")
    for i in range(1, len(poses)):
        a, b = poses[i-1], poses[i]
        fractions = []
        for axis in range(2):
            delta = b[axis]-a[axis]
            if delta > 0 and b[axis] >= upper[axis]:
                fractions.append((upper[axis]-a[axis])/delta)
            elif delta < 0 and b[axis] <= lower[axis]:
                fractions.append((lower[axis]-a[axis])/delta)
        if not fractions:
            continue
        alpha = min(fractions)
        if not 0. <= alpha <= 1.:
            raise ValueError("invalid GT crop intersection")
        end = np.r_[a[:2]+alpha*(b[:2]-a[:2]),
                    float(wrap_angle(a[2]+alpha*float(wrap_angle(b[2]-a[2]))))]
        end_time = times[i-1]+alpha*(times[i]-times[i-1])
        prefix = np.vstack([poses[:i], end])
        prefix_times = np.r_[times[:i], end_time]
        return dict(poses=prefix, times=prefix_times, boundary_inset_m=float(inset_m),
                    end_interpolated=bool(alpha < 1.-1e-8), boundary_segment_index=i-1,
                    boundary_fraction=float(alpha), available_horizon_s=float(times[-1]),
                    source="logged_future_prefix_with_boundary_intersection")
    return None


def geometric_gt(poses):
    """Remove identical stationary poses only for static geometry checking.

    The saved logged trajectory and timestamps are never deduplicated.
    """
    poses = np.asarray(poses)
    keep = np.r_[True, (np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1) > 1e-7)
                 | (np.abs(wrap_angle(np.diff(poses[:, 2]))) > 1e-7)]
    return poses[keep]
