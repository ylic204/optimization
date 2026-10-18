"""Tilebased v85 contract and exact solver (NumPy only; no Qwen/CUDA needed).

Upstream eight-neighbor node-cost graph: all finite cells are traversable,
including Pokemon wall=25; diagonals have no sqrt(2) multiplier. Reporting
includes both endpoints, as sum(costs * path_mask). Including the source
adds the same constant to every route and preserves the upstream optimum.
"""
import heapq
import json
import numpy as np

CONTRACT = "tilebased_mincost_v85_r1"
MODEL_SCHEMA = "tilebased_node_cost_path_v1"
UPSTREAM_REV = "384900fd82d8404a6eb0deb326412755620f3747"
DATASETS = {
    "pkmn": {"grid": 20, "image_side": 320, "region_grid": 10,
             "cost_values": [1., 1.5, 1.8, 2., 2.5, 3.5, 6., 8., 10., 25.]},
    "warcraft": {"grid": 12, "image_side": 96, "region_grid": 3,
                 "cost_values": [.8, 1.2, 5.3, 7.7, 9.2]},
}
PKMN_LEGEND = ("sand/stairs=1; grass=1.5; snow=1.8; dirt=2; leaves=2.5; "
               "plants=3.5; lake=6; mud=8; ocean=10; wall=25. "
               "Wall is a finite high-cost cell under the original benchmark rules.")


def coordinate(value, shape):
    a = np.asarray(value)
    if a.shape != (2,) or a.dtype.kind not in 'iu' or not ((a >= 0) & (a < shape)).all():
        raise ValueError("coordinates must be zero-based integer [row,column] inside the grid")
    return tuple(map(int, a))


def neighbors(node, shape):
    r, c = node
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            v = r + dr, c + dc
            if (dr or dc) and 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]:
                yield v


def shortest_path(costs, source, target, allowed=None):
    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 2 or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("costs must be a positive finite matrix")
    source, target = coordinate(source, costs.shape), coordinate(target, costs.shape)
    allowed = np.ones(costs.shape, bool) if allowed is None else np.asarray(allowed, bool)
    if allowed.shape != costs.shape:
        raise ValueError("allowed mask shape mismatch")
    if not allowed[source] or not allowed[target]:
        return None
    distances = {source: float(costs[source])}
    parents = {source: None}
    queue = [(distances[source], source)]
    while queue:
        value, node = heapq.heappop(queue)
        if value != distances[node]:
            continue
        if node == target:
            path = []
            while node is not None:
                path.append(node)
                node = parents[node]
            return np.asarray(path[::-1], dtype=np.int64)
        for nxt in neighbors(node, costs.shape):
            if not allowed[nxt]:
                continue
            distance = value + float(costs[nxt])
            if distance < distances.get(nxt, float('inf')):
                distances[nxt], parents[nxt] = distance, node
                heapq.heappush(queue, (distance, nxt))
    return None


def path_cost(costs, path):
    p = np.asarray(path, dtype=np.int64)
    return float(np.asarray(costs, dtype=np.float64)[p[:, 0], p[:, 1]].sum())


def validate_path(path, source, target, shape):
    a = np.asarray(path)
    if a.ndim != 2 or a.shape[1] != 2 or not len(a) or a.dtype.kind not in 'iu':
        raise ValueError("path must be a nonempty integer [N,2] array")
    for p in a:
        coordinate(p, shape)
    if tuple(a[0]) != tuple(source) or tuple(a[-1]) != tuple(target):
        raise ValueError("path endpoints mismatch")
    if len(a) > 1 and not (np.abs(np.diff(a, axis=0)).max(axis=1) == 1).all():
        raise ValueError("path contains non-neighbor moves")
    if len(set(map(tuple, a))) != len(a):
        raise ValueError("path has repeated nodes")


def verify_gt(costs, mask, source, target):
    mask = np.asarray(mask)
    if mask.shape != costs.shape or not np.isin(mask, [0, 1]).all():
        raise ValueError("GT must be a binary path mask")
    ordered = shortest_path(costs, source, target, mask.astype(bool))
    if ordered is None or len(ordered) != int(mask.sum()):
        raise ValueError("GT mask is disconnected or contains extra nodes")
    validate_path(ordered, source, target, costs.shape)
    exact = shortest_path(costs, source, target)
    optimum = path_cost(costs, exact)
    if not np.isclose(path_cost(costs, ordered), optimum, rtol=1e-5, atol=1e-5):
        raise ValueError("original GT is not optimal under the declared graph convention")
    return ordered, optimum


def task_text(dataset, source, target):
    n = DATASETS[dataset]['grid']
    return (f"Find a minimum total terrain-cost path on this {n}x{n} map from "
            f"S={list(map(int, source))} to G={list(map(int, target))}. "
            "Coordinates are zero-based [row,column], row down and column right. "
            "Use eight-neighbor moves including diagonals, with no diagonal multiplier. "
            "Count source and target costs. All cells have finite positive costs. "
            "No step, time or energy constraint.")


def build_model_prompt(task, dataset, source, target, retained_indices, region_grid, legend):
    """Allowlisted PUBLIC arguments only. Never accept a full dataset batch."""
    n = DATASETS[dataset]['grid']
    return (f"{task}\nBuild a minimum-cost path optimization instance; return ONE JSON object. "
            "The solver uses binary arc choices with source/target flow balance on the fixed "
            "eight-neighbor graph, objective sum of visited node costs. Infer node costs from "
            "the image; do not output a route. Missing regions are unknown, not free. "
            "Complete the grid using your visual estimates; no hidden true costs are available. "
            f"Public terrain legend: {legend}\n"
            f"Retained visual regions on a {region_grid}x{region_grid} row-major grid: "
            f"{list(map(int, retained_indices))}. These are visual regions, not terrain cells. "
            f"Return exactly: {{\"schema\":\"{MODEL_SCHEMA}\","
            f"\"source_rc\":{list(map(int, source))},\"target_rc\":{list(map(int, target))},"
            f"\"node_costs\":[...]}}. node_costs must be {n} rows of {n} finite positive numbers "
            f"from {DATASETS[dataset]['cost_values']}. No extra keys or executable code.")


def validate_model(answer, dataset, source, target):
    text = str(answer).strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    def unique_keys(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError('duplicate JSON key')
            result[k] = v
        return result
    obj = json.loads(text, object_pairs_hook=unique_keys)
    if not isinstance(obj, dict) or set(obj) != {'schema', 'source_rc', 'target_rc', 'node_costs'}:
        raise ValueError('incorrect optimization schema fields')
    n = DATASETS[dataset]['grid']
    if obj['schema'] != MODEL_SCHEMA:
        raise ValueError('incorrect optimization schema version')
    if coordinate(obj['source_rc'], (n, n)) != tuple(source) or coordinate(obj['target_rc'], (n, n)) != tuple(target):
        raise ValueError('model changed the task endpoints')
    costs = np.asarray(obj['node_costs'])
    if not isinstance(obj['node_costs'], list) or any(
            not isinstance(row, list) or any(type(x) not in (int, float) for x in row)
            for row in obj['node_costs']):
        raise ValueError('node_costs must contain JSON numbers, not strings or booleans')
    if costs.shape != (n, n) or costs.dtype.kind not in 'fiu' or not np.isfinite(costs).all():
        raise ValueError('node_costs has incorrect shape/type/nonfinite values')
    allowed = np.asarray(DATASETS[dataset]['cost_values'])
    if not np.isclose(costs[..., None], allowed, atol=1e-5, rtol=1e-5).any(-1).all():
        raise ValueError('node_costs contains values outside the public cost vocabulary')
    return costs.astype(np.float64)


def evaluate_path(path, true_costs, source, target, optimal_cost):
    validate_path(path, source, target, true_costs.shape)
    cost = path_cost(true_costs, path)
    regret = max(0., (cost - float(optimal_cost)) / max(float(optimal_cost), 1e-8))
    return {'true_cost': cost, 'optimal_cost': float(optimal_cost),
            'relative_regret': regret, 'optimal': bool(np.isclose(cost, optimal_cost, atol=1e-5, rtol=1e-5)),
            'path_rc': np.asarray(path).tolist()}
