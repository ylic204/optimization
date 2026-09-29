"""Preview generator for a sensor-style local occupancy/BEV cost map.

This prototype deliberately does not simulate a robot body, camera pose, goal
icon, or individual LiDAR rays.  It directly generates the local map that a
mapping front-end could provide.  The model input contains only occupancy and
terrain/risk layers; candidate paths and the optimum are debug/GT fields.
"""

import argparse
import heapq
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CELL_GRID = 72
SELECTOR_GRID = 9
CELLS_PER_PATCH = CELL_GRID // SELECTOR_GRID
N_CANDIDATE_PATHS = 16

FREE = 0
OCCUPIED = 1
UNKNOWN = 2

NORMAL = 0
ROUGH = 1
HAZARD = 2

STATE_COLORS = {
    "free": (225, 226, 224),
    "occupied": (48, 52, 55),
    "unknown": (139, 143, 145),
    "rough": (190, 151, 83),
    "hazard": (184, 76, 67),
}


def _ellipse_mask(rows, columns, center_row, center_column, radius_r, radius_c):
    rr, cc = np.ogrid[:rows, :columns]
    return (
        ((rr - center_row) / max(radius_r, 1)) ** 2
        + ((cc - center_column) / max(radius_c, 1)) ** 2
    ) <= 1.0


def _make_layers(rng):
    occupancy = np.full((CELL_GRID, CELL_GRID), FREE, dtype=np.int8)
    terrain = np.full_like(occupancy, NORMAL)

    # Local structural map: boundary walls plus staggered internal walls and
    # door-like gaps.  No graph lines are drawn into the model input.
    occupancy[[0, -1], :] = OCCUPIED
    for column in (12, 24, 36, 48, 60):
        occupancy[:, column : column + 2] = OCCUPIED
        gap_centers = rng.choice(np.arange(9, 64), size=3, replace=False)
        for center in gap_centers:
            lo = max(1, int(center) - 4)
            hi = min(CELL_GRID - 1, int(center) + 5)
            occupancy[lo:hi, column : column + 2] = FREE

    # Short rectangular obstacles make the map look like a local costmap
    # rather than a hand-drawn candidate graph.
    for _ in range(12):
        row = int(rng.integers(4, CELL_GRID - 12))
        column = int(rng.integers(3, CELL_GRID - 10))
        height = int(rng.integers(3, 9))
        width = int(rng.integers(3, 8))
        occupancy[row : row + height, column : column + width] = OCCUPIED

    # Semantic traversal-cost regions are spatially coherent blobs.
    for state, count in ((ROUGH, 9), (HAZARD, 6)):
        for _ in range(count):
            center_row = int(rng.integers(5, CELL_GRID - 5))
            center_column = int(rng.integers(5, CELL_GRID - 5))
            mask = _ellipse_mask(
                CELL_GRID,
                CELL_GRID,
                center_row,
                center_column,
                int(rng.integers(3, 8)),
                int(rng.integers(3, 9)),
            )
            terrain[mask & (occupancy == FREE)] = state

    # Unknown areas are map-layer values, not simulated LiDAR rays.  They are
    # assigned only to geometrically free cells and receive a planning penalty.
    unknown = np.zeros_like(occupancy, dtype=bool)
    for _ in range(7):
        row = int(rng.integers(2, CELL_GRID - 14))
        column = int(rng.integers(2, CELL_GRID - 14))
        height = int(rng.integers(5, 13))
        width = int(rng.integers(5, 13))
        unknown[row : row + height, column : column + width] = True
    unknown &= occupancy == FREE

    start = (CELL_GRID // 2, 1)
    goal = (CELL_GRID // 2, CELL_GRID - 2)
    occupancy[start] = FREE
    occupancy[goal] = FREE
    unknown[start] = False
    unknown[goal] = False
    return occupancy, terrain, unknown, start, goal


def _astar(occupancy, cell_cost, start, goal):
    moves = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 2**0.5),
        (-1, 1, 2**0.5),
        (1, -1, 2**0.5),
        (1, 1, 2**0.5),
    )
    queue = [(0.0, start)]
    distance = {start: 0.0}
    predecessor = {}
    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            break
        current_distance = distance[current]
        for dr, dc, step_length in moves:
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < CELL_GRID and 0 <= nxt[1] < CELL_GRID):
                continue
            if occupancy[nxt] == OCCUPIED:
                continue
            candidate = current_distance + step_length * float(cell_cost[nxt])
            if candidate < distance.get(nxt, float("inf")):
                distance[nxt] = candidate
                predecessor[nxt] = current
                heuristic = ((nxt[0] - goal[0]) ** 2 + (nxt[1] - goal[1]) ** 2) ** 0.5
                heapq.heappush(queue, (candidate + heuristic, nxt))
    if goal not in distance:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(predecessor[path[-1]])
    path.reverse()
    return np.asarray(path, dtype=np.int16)


def _path_cost(path, true_cost):
    delta = np.diff(path.astype(np.float32), axis=0)
    step = np.sqrt((delta**2).sum(axis=1))
    return float((step * true_cost[path[1:, 0], path[1:, 1]]).sum())


def _candidate_paths(rng, occupancy, terrain, unknown, start, goal):
    true_cost = np.ones_like(occupancy, dtype=np.float32)
    true_cost += (terrain == ROUGH) * 0.55
    true_cost += (terrain == HAZARD) * 2.00
    true_cost += unknown * 0.85

    candidates = []
    signatures = set()
    reuse_penalty = np.zeros_like(true_cost)
    for attempt in range(96):
        perturbed = true_cost.copy()
        perturbed += reuse_penalty * (0.12 + 0.04 * (attempt % 5))
        perturbed += rng.uniform(0.0, 0.20, size=perturbed.shape).astype(np.float32)
        path = _astar(occupancy, perturbed, start, goal)
        if path is None:
            continue
        signature = tuple(map(tuple, path.tolist()))
        if signature not in signatures:
            signatures.add(signature)
            candidates.append(path)
            reuse_penalty[path[:, 0], path[:, 1]] += 1.0
        if len(candidates) == N_CANDIDATE_PATHS:
            break
    if not candidates:
        raise RuntimeError("generated BEV map has no feasible west-east path")

    path_cost = np.asarray(
        [_path_cost(path, true_cost) for path in candidates], dtype=np.float32
    )
    optimal_index = int(np.argmin(path_cost))
    path_patch_mask = np.zeros(
        (len(candidates), SELECTOR_GRID**2), dtype=np.float32
    )
    for path_index, path in enumerate(candidates):
        patch_rows = path[:, 0] // CELLS_PER_PATCH
        patch_columns = path[:, 1] // CELLS_PER_PATCH
        patch_ids = np.unique(patch_rows * SELECTOR_GRID + patch_columns)
        path_patch_mask[path_index, patch_ids] = 1.0
    return candidates, path_cost, optimal_index, path_patch_mask, true_cost


def render_bev(occupancy, terrain, unknown, scale=4):
    rgb = np.empty((CELL_GRID, CELL_GRID, 3), dtype=np.uint8)
    rgb[:] = STATE_COLORS["free"]
    rgb[terrain == ROUGH] = STATE_COLORS["rough"]
    rgb[terrain == HAZARD] = STATE_COLORS["hazard"]
    rgb[unknown] = STATE_COLORS["unknown"]
    rgb[occupancy == OCCUPIED] = STATE_COLORS["occupied"]
    return np.asarray(
        Image.fromarray(rgb).resize(
            (CELL_GRID * scale, CELL_GRID * scale), Image.Resampling.NEAREST
        )
    )


def build_one(rng):
    for _ in range(100):
        occupancy, terrain, unknown, start, goal = _make_layers(rng)
        try:
            paths, costs, optimal, patch_mask, true_cost = _candidate_paths(
                rng, occupancy, terrain, unknown, start, goal
            )
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("failed to generate a feasible BEV sample")

    image = render_bev(occupancy, terrain, unknown)
    max_length = max(len(path) for path in paths)
    padded_paths = np.full((len(paths), max_length, 2), -1, dtype=np.int16)
    path_lengths = np.zeros(len(paths), dtype=np.int16)
    for index, path in enumerate(paths):
        padded_paths[index, : len(path)] = path
        path_lengths[index] = len(path)
    return {
        "image": image,
        "occupancy": occupancy,
        "terrain": terrain,
        "unknown": unknown.astype(np.int8),
        "true_cell_cost": true_cost,
        "candidate_paths": padded_paths,
        "candidate_path_lengths": path_lengths,
        "candidate_path_costs": costs,
        "candidate_path_patch_mask": patch_mask,
        "optimal_path_idx": np.int64(optimal),
        "optimal_cost": np.float32(costs[optimal]),
        "selector_grid": np.int64(SELECTOR_GRID),
        # Fixed boundary portals are optimization metadata, not rendered robot
        # or target objects in the model input.
        "planning_start": np.asarray(start, dtype=np.int16),
        "planning_goal": np.asarray(goal, dtype=np.int16),
        "layout": np.asarray("local_occupancy_bev_v1"),
    }


def render_preview(sample, output):
    input_image = Image.fromarray(sample["image"])
    width = input_image.width
    gap = 28
    top = 42
    canvas = Image.new("RGB", (3 * width + 2 * gap, width + 92), (246, 246, 244))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 15) if Path(font_path).exists() else None
    small = ImageFont.truetype(font_path, 11) if Path(font_path).exists() else None
    draw = ImageDraw.Draw(canvas)

    canvas.paste(input_image, (0, top))
    draw.text((0, 10), "A. Model input: local occupancy / BEV", fill=(20, 20, 20), font=font)

    patch_view = input_image.copy()
    patch_draw = ImageDraw.Draw(patch_view)
    patch_side = width // SELECTOR_GRID
    for index in range(SELECTOR_GRID + 1):
        coordinate = index * patch_side
        patch_draw.line((coordinate, 0, coordinate, width), fill=(45, 102, 180), width=1)
        patch_draw.line((0, coordinate, width, coordinate), fill=(45, 102, 180), width=1)
    for row in range(SELECTOR_GRID):
        for column in range(SELECTOR_GRID):
            patch_draw.text(
                (column * patch_side + 2, row * patch_side + 1),
                str(row * SELECTOR_GRID + column),
                fill=(20, 55, 100),
                font=small,
            )
    x2 = width + gap
    canvas.paste(patch_view, (x2, top))
    draw.text((x2, 10), "B. Qwen selector regions (9 x 9)", fill=(20, 20, 20), font=font)

    debug = input_image.copy()
    debug_draw = ImageDraw.Draw(debug)
    scale = width / CELL_GRID
    lengths = sample["candidate_path_lengths"]
    for index in range(min(8, len(lengths))):
        path = sample["candidate_paths"][index, : lengths[index]]
        points = [
            (int((column + 0.5) * scale), int((row + 0.5) * scale))
            for row, column in path
        ]
        debug_draw.line(points, fill=(82, 116, 166), width=1)
    optimum = int(sample["optimal_path_idx"])
    path = sample["candidate_paths"][optimum, : lengths[optimum]]
    points = [
        (int((column + 0.5) * scale), int((row + 0.5) * scale))
        for row, column in path
    ]
    debug_draw.line(points, fill=(35, 190, 82), width=4)
    x3 = 2 * (width + gap)
    canvas.paste(debug, (x3, top))
    draw.text((x3, 10), "C. Planning GT (debug only)", fill=(20, 20, 20), font=font)

    legend_y = top + width + 20
    legend_items = (
        ("free", STATE_COLORS["free"]),
        ("occupied", STATE_COLORS["occupied"]),
        ("unknown", STATE_COLORS["unknown"]),
        ("rough", STATE_COLORS["rough"]),
        ("hazard", STATE_COLORS["hazard"]),
        ("optimal path", (35, 190, 82)),
    )
    legend_x = 0
    for label, color in legend_items:
        draw.rectangle(
            (legend_x, legend_y, legend_x + 13, legend_y + 13), fill=color
        )
        draw.text(
            (legend_x + 18, legend_y - 1), label, fill=(35, 35, 35), font=small
        )
        legend_x += 105 if label != "optimal path" else 130
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bev_v80_preview.npz")
    parser.add_argument("--preview", default="bev_v80_preview.png")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    sample = build_one(np.random.default_rng(args.seed))
    np.savez_compressed(args.out, **sample)
    render_preview(sample, args.preview)
    print(f"saved sample: {args.out}")
    print(f"saved preview: {args.preview}")
    print(
        f"candidate_paths={len(sample['candidate_path_lengths'])}, "
        f"optimal={int(sample['optimal_path_idx'])}, "
        f"cost={float(sample['optimal_cost']):.3f}"
    )


if __name__ == "__main__":
    main()
