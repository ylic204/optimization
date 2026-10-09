"""Frozen BEV modeling -> validated continuous QP -> external trajectory reward.

Only retained vision tokens, task text, geometry, current ego and goal enter
the model. GT and semantic occupancy labels enter evaluation after solving.
The JSON grammar is a bounded model family, not executable generated code.
"""

import json
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, minimize

from bev_geometry_v85 import sample_bev_grid_nearest
from trajectory_gt_v85 import displacement_errors


MODEL_SCHEMA = "bev_trajectory_qp_v1"
ENVIRONMENT_CONTRACT = "frozen_bev_llm_continuous_qp_gt_v1"
SELECTOR_TEXT_CONTRACT = "task_model_requirements_v1"
REQUIREMENTS = (
    "Choose BEV evidence for a trajectory QP: route/reference, drivable corridor, "
    "actors and clearance, traffic stop boundaries. Variables: future rear-axle xy; "
    "tracking/smoothness/goal cost, corridor and separating halfspaces, fixed dynamics."
)


def selector_requirement_text(task_text):
    return f"Task: {task_text}\n{REQUIREMENTS}"


@dataclass(frozen=True)
class SolverConfig:
    max_speed: float = 15.0
    max_acceleration: float = 4.0
    max_iterations: int = 300
    feasibility_tolerance: float = 1e-4

    def __post_init__(self):
        if min(self.max_speed, self.max_acceleration, self.feasibility_tolerance) <= 0 or self.max_iterations <= 0:
            raise ValueError("solver bounds, tolerance and iterations must be positive")


def build_model_prompt(task_text, times, bounds, ego, goal):
    """Explicit allowlist: no batch dictionary or privileged annotations."""
    return (
        "Build a continuous trajectory optimization instance from the visible BEV. "
        "Return ONE JSON object only. x is forward, y is left, origin is current rear axle. "
        "BEV rows run x_max to x_min; columns run y_max to y_min. "
        "Colors: light blue=drivable, darker light blue=route, purple=reference, "
        "blue/brown/pink/black=actors, green trails=history, red/yellow/green=signals, "
        "orange=ego, gold=goal. Missing regions are unknown. Do not treat them as free space. "
        "Plan conservative corridors; account for ego footprint and current actors in corridor/halfspaces. "
        "The template has fixed initial position, measured initial velocity, speed/acceleration bounds. "
        "You supply visual reference, corridor and separating/stop constraints, and objective weights. "
        "This is a convex point-trajectory approximation, not bicycle dynamics. "
        f"Task: {task_text}\nTimes seconds: {list(map(float, times))}\n"
        f"BEV bounds [xmin,xmax,ymin,ymax]: {list(map(float, bounds))}\n"
        f"Current velocity xy: {list(map(float, ego[:2]))}; goal xy: {list(map(float, goal[:2]))}\n"
        f"Schema: {{\"schema\":\"{MODEL_SCHEMA}\",\"reference_xy\":[[x,y],...],"
        "\"corridor\":[[xmin,xmax,ymin,ymax],...],"
        "\"weights\":{\"tracking\":1,\"smoothness\":1,\"goal\":1},"
        "\"halfspaces\":[{\"step\":1,\"normal\":[nx,ny],\"upper\":b},...]}. "
        f"reference_xy and corridor each have exactly {len(times)} future rows, in time order. "
        "Corridors must lie inside BEV bounds and have positive width. "
        "Weights: tracking in [0.001,100], smoothness/goal in [0,100]. "
        f"Halfspace step is 1..{len(times)}, normal dot xy <= upper; at most 64. "
        "Use an empty halfspaces list when none is needed. Do not output extra fields."
    )


def _finite_array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def validate_model(answer, times, bounds):
    text = str(answer).strip()
    # Markdown wrappers are tolerated; multiple/trailing objects are rejected.
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    payload = json.loads(text)
    if set(payload) != {"schema", "reference_xy", "corridor", "weights", "halfspaces"} or payload["schema"] != MODEL_SCHEMA:
        raise ValueError("unsupported or incomplete optimization model schema")
    n = len(times)
    reference = _finite_array(payload["reference_xy"], (n, 2), "reference_xy")
    corridor = _finite_array(payload["corridor"], (n, 4), "corridor")
    bounds = _finite_array(bounds, (4,), "map bounds")
    if (corridor[:, 0] >= corridor[:, 1]).any() or (corridor[:, 2] >= corridor[:, 3]).any():
        raise ValueError("corridor must have positive widths")
    if ((corridor[:, [0, 2]] < bounds[[0, 2]] - 1e-6).any()
            or (corridor[:, [1, 3]] > bounds[[1, 3]] + 1e-6).any()):
        raise ValueError("corridor exceeds BEV support")
    if (reference < bounds[[0, 2]]).any() or (reference > bounds[[1, 3]]).any():
        raise ValueError("reference exceeds BEV support")
    weights = payload["weights"]
    if set(weights) != {"tracking", "smoothness", "goal"}:
        raise ValueError("unknown objective weights")
    w = _finite_array([weights[x] for x in ("tracking", "smoothness", "goal")], (3,), "weights")
    if w[0] < 0.001 or (w < 0).any() or (w > 100).any():
        raise ValueError("weights must preserve a bounded convex objective")
    halfspaces = payload["halfspaces"]
    if not isinstance(halfspaces, list) or len(halfspaces) > 64:
        raise ValueError("at most 64 halfspaces are accepted")
    for h in halfspaces:
        if set(h) != {"step", "normal", "upper"} or type(h["step"]) is not int or not 1 <= h["step"] <= n:
            raise ValueError("invalid halfspace index or keys")
        normal = _finite_array(h["normal"], (2,), "halfspace normal")
        if np.linalg.norm(normal) < 1e-8 or not np.isfinite(float(h["upper"])):
            raise ValueError("invalid separating halfspace")
        h["normal"], h["upper"] = normal.tolist(), float(h["upper"])
    payload["reference_xy"], payload["corridor"] = reference.tolist(), corridor.tolist()
    payload["weights"] = dict(zip(("tracking", "smoothness", "goal"), w.tolist()))
    return payload


def solve_trajectory_qp(payload, times, bounds, ego, goal, config):
    """Optimize all N continuous xy states; no candidate selection or GT input."""
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or (np.diff(np.r_[0., times]) <= 0).any():
        raise ValueError("invalid future time grid")
    payload = validate_model(json.dumps(payload), times, bounds)
    n = len(times)
    dt = np.diff(np.r_[0., times])
    # V p gives per-axis average velocity, A p + a0 gives acceleration.
    difference = np.eye(n) - np.eye(n, k=-1)
    v = np.kron(difference / dt[:, None], np.eye(2))
    av = np.kron(difference / dt[:, None], np.eye(2))
    acceleration = av @ v
    a0 = np.zeros(2 * n)
    a0[:2] = -np.asarray(ego, dtype=np.float64)[:2] / dt[0]
    reference = np.asarray(payload["reference_xy"], dtype=np.float64).ravel()
    wt, ws, wg = [float(payload["weights"][k]) for k in ("tracking", "smoothness", "goal")]
    terminal = np.zeros((2, 2 * n))
    terminal[:, -2:] = np.eye(2)
    hessian = 2 * (wt * np.eye(2 * n) / n + ws * acceleration.T @ acceleration / n + wg * terminal.T @ terminal)
    linear = -2 * wt * reference / n + 2 * ws * acceleration.T @ a0 / n - 2 * wg * terminal.T @ np.asarray(goal)[:2]
    corridor = np.asarray(payload["corridor"], dtype=np.float64)
    lower, upper = corridor[:, [0, 2]].ravel(), corridor[:, [1, 3]].ravel()
    matrices = [v, acceleration]
    lows = [np.full(2 * n, -config.max_speed), -config.max_acceleration - a0]
    highs = [np.full(2 * n, config.max_speed), config.max_acceleration - a0]
    for halfspace in payload["halfspaces"]:
        row = np.zeros((1, 2 * n))
        norm = np.linalg.norm(halfspace["normal"])
        offset = 2 * (halfspace["step"] - 1)
        row[0, offset:offset + 2] = np.asarray(halfspace["normal"]) / norm
        matrices.append(row)
        lows.append(np.asarray([-np.inf]))
        highs.append(np.asarray([halfspace["upper"] / norm]))
    matrix, lo, hi = np.vstack(matrices), np.concatenate(lows), np.concatenate(highs)
    result = minimize(
        lambda p: 0.5 * p @ hessian @ p + linear @ p,
        np.clip(reference, lower, upper), jac=lambda p: hessian @ p + linear,
        method="SLSQP", bounds=Bounds(lower, upper),
        constraints=[LinearConstraint(matrix, lo, hi)],
        options={"maxiter": config.max_iterations, "ftol": 1e-8},
    )
    if result.x is None or not np.isfinite(result.x).all():
        return {"success": False, "message": str(result.message), "max_violation": None}
    violation = float(max(0., np.max(lower - result.x), np.max(result.x - upper),
                          np.max(lo - matrix @ result.x), np.max(matrix @ result.x - hi)))
    success = bool(result.success and violation <= config.feasibility_tolerance)
    output = {"success": success, "message": str(result.message), "max_violation": violation,
              "iterations": int(result.nit)}
    if success:
        xy = result.x.reshape(n, 2)
        delta = np.diff(np.vstack([np.zeros((1, 2)), xy]), axis=0)
        output["trajectory"] = np.c_[xy, np.arctan2(delta[:, 1], delta[:, 0])].tolist()
    return output


class FrozenBEVSolverEnvironment:
    def __init__(self, backbone, args):
        self.backbone = backbone
        self.config = SolverConfig(args.solver_max_speed, args.solver_max_acceleration,
                                   args.solver_max_iterations)
        self.max_new_tokens = args.modeling_max_new_tokens
        self.ade_weight, self.fde_weight = args.rl_ade_weight, args.rl_fde_weight
        self.safety_weight, self.failure_cost = args.rl_violation_weight, args.rl_failure_cost

    @torch.no_grad()
    def evaluate(self, encoded, indices, batch, texts):
        """One call per fixed subset; return failure costs rather than fallbacks."""
        indices = indices.sort(dim=-1).values
        results = []
        for row in range(len(indices)):
            def array(key):
                value = batch[key][row]
                return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            if not bool(array("trajectory_gt_available")) or bool(array("expert_is_fallback")):
                raise ValueError("solver RL/evaluation requires recorded, time-aligned future GT")
            times, bounds, ego, goal = [array(k) for k in ("planning_time_s", "map_bounds", "ego_state", "goal_state")]
            prompt = build_model_prompt(texts[row], times, bounds, ego, goal)
            start = time.perf_counter()
            single = {key: value[row:row + 1] for key, value in encoded["vision_state"].items()
                      if key != "position_groups"}
            if "position_groups" in encoded["vision_state"]:
                single["position_groups"] = tuple(x[row:row + 1] for x in encoded["vision_state"]["position_groups"])
            item = {"status": "model_failure", "loss": self.failure_cost,
                    "ade_m": None, "fde_m": None, "collision_proxy": None,
                    "non_traversable_proxy": None, "trajectory": None,
                    "selected_indices": indices[row].cpu().tolist(), "time_s": times.tolist()}
            try:
                answer = self.backbone.generate_selected_json(single, indices[row:row + 1], prompt, self.max_new_tokens)
                item["model_answer"] = answer
                payload = validate_model(answer, times, bounds)
                item["optimization_model"] = payload
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                item["error"] = str(exc)
            else:
                solved = solve_trajectory_qp(payload, times, bounds, ego, goal, self.config)
                item["solver"] = {k: v for k, v in solved.items() if k != "trajectory"}
                item["status"] = "solver_failure"
                if solved["success"]:
                    trajectory = np.asarray(solved["trajectory"])
                    ade, fde = displacement_errors(trajectory[:, :2], array("expert_trajectory"))
                    obstacle, inside = sample_bev_grid_nearest(array("dynamic_obstacle"), trajectory[:, :2], bounds)
                    drivable, _ = sample_bev_grid_nearest(array("traversable"), trajectory[:, :2], bounds)
                    collision = float((obstacle & inside).mean())
                    offroad = float((~drivable | ~inside).mean())
                    # Uniform bounded loss: a failure is never cheaper than a
                    # successful plan. ADE/FDE remain unclipped in reports.
                    loss = self.ade_weight * ade + self.fde_weight * fde + self.safety_weight * (collision + offroad)
                    item.update(status="success", trajectory=solved["trajectory"], ade_m=ade, fde_m=fde,
                                collision_proxy=collision, non_traversable_proxy=offroad,
                                loss=min(float(loss), self.failure_cost * 0.999))
            item["environment_seconds"] = time.perf_counter() - start
            results.append(item)
        return results


def environment_losses(results, reference):
    return reference.new_tensor([x["loss"] for x in results])


def summarize_outcomes(results):
    successful = [x for x in results if x["status"] == "success"]
    summary = {"samples": len(results), "successful_samples": len(successful),
               "task_loss": float(np.mean([x["loss"] for x in results])),
               "model_failure_rate": float(np.mean([x["status"] == "model_failure" for x in results])),
               "solver_failure_rate": float(np.mean([x["status"] == "solver_failure" for x in results])),
               "environment_seconds": float(np.mean([x["environment_seconds"] for x in results]))}
    for key in ("ade_m", "fde_m", "collision_proxy", "non_traversable_proxy"):
        summary[key] = float(np.mean([x[key] for x in successful])) if successful else None
    return summary
