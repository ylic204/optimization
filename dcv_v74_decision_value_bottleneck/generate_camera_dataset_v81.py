"""Generate first-person camera observations for local trajectory selection.

The student observation is a single RGB image.  Depth, semantic segmentation,
the local BEV and camera-to-BEV correspondences are privileged targets: they
may supervise training and diagnostics, but they are not required at student
inference time.

The renderer is intentionally lightweight.  It builds a procedural 2.5-D
world, extrudes occupied BEV cells into vertical obstacles, and ray-casts an
RGB-D/semantic camera view.  This is a controlled token-pruning benchmark, not
a photorealistic replacement for Habitat, Isaac Sim or Gazebo.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WORLD_GRID = 72
IMAGE_SIDE = 288
SELECTOR_GRID = 9
PATCH_SIDE = IMAGE_SIDE // SELECTOR_GRID
CELL_SIZE_M = 0.25
CAMERA_HEIGHT_M = 1.05
OBSTACLE_HEIGHT_M = 1.8
MAX_DEPTH_M = 16.0
HORIZONTAL_FOV_DEG = 72.0
VERTICAL_FOV_DEG = 58.0
CAMERA_PITCH_DEG = -9.0
N_TRAJECTORIES = 11
TRAJECTORY_SAMPLES = 64
TRAJECTORY_HORIZON_M = 12.0

FREE = 0
OCCUPIED = 1

NORMAL = 0
ROUGH = 1
HAZARD = 2

SEM_SKY = 0
SEM_FLOOR = 1
SEM_ROUGH = 2
SEM_HAZARD = 3
SEM_OBSTACLE = 4

SEMANTIC_COLORS = np.asarray(
    [
        (105, 155, 210),  # sky
        (156, 164, 151),  # normal traversable floor
        (176, 128, 63),   # rough terrain
        (194, 61, 55),    # hazardous terrain
        (50, 55, 60),     # occupied obstacle
    ],
    dtype=np.uint8,
)

DEFAULT_TASK_TEXT = (
    "Choose the safest low-cost local trajectory through the visible scene. "
    "Avoid obstacles and hazardous terrain while maintaining forward progress."
)


def _ellipse_mask(rows, columns, center_row, center_column, radius_r, radius_c):
    rr, cc = np.ogrid[:rows, :columns]
    return (
        ((rr - center_row) / max(radius_r, 1)) ** 2
        + ((cc - center_column) / max(radius_c, 1)) ** 2
    ) <= 1.0


def _make_world(rng):
    occupancy = np.full((WORLD_GRID, WORLD_GRID), FREE, dtype=np.int8)
    terrain = np.full_like(occupancy, NORMAL)

    # Boundary and scene obstacles.  These are BEV geometry only; the model
    # never receives this array as its visual observation.
    occupancy[[0, -1], :] = OCCUPIED
    occupancy[:, [0, -1]] = OCCUPIED
    for _ in range(15):
        row = int(rng.integers(10, 59))
        column = int(rng.integers(4, 65))
        height = int(rng.integers(2, 8))
        width = int(rng.integers(2, 8))
        occupancy[row : row + height, column : column + width] = OCCUPIED

    # Short longitudinal wall segments produce occlusion without forcing all
    # fixed motion primitives through several mutually incompatible doors.
    for column in rng.choice(np.arange(12, 61), size=4, replace=False):
        start = int(rng.integers(17, 48))
        length = int(rng.integers(7, 16))
        occupancy[start : start + length, column : column + 2] = OCCUPIED

    for state, count in ((ROUGH, 8), (HAZARD, 6)):
        for _ in range(count):
            mask = _ellipse_mask(
                WORLD_GRID,
                WORLD_GRID,
                int(rng.integers(10, 63)),
                int(rng.integers(5, 67)),
                int(rng.integers(3, 8)),
                int(rng.integers(3, 9)),
            )
            terrain[mask & (occupancy == FREE)] = state

    robot_cell = np.asarray([65, 36], dtype=np.int16)
    # The camera pose is fixed benchmark metadata, not a learned graph
    # feature.  Clear only the immediate footprint so scenes remain varied.
    occupancy[62:69, 32:41] = FREE
    terrain[62:69, 32:41] = NORMAL
    return occupancy, terrain, robot_cell


def _motion_primitives(robot_cell):
    """Return canonically ordered left-to-right local trajectories."""
    forward = np.linspace(0.0, TRAJECTORY_HORIZON_M, TRAJECTORY_SAMPLES)
    offsets = np.linspace(-4.25, 4.25, N_TRAJECTORIES)
    trajectories_robot = []
    trajectories_cells = []
    u = forward / TRAJECTORY_HORIZON_M
    blend = u * u * (3.0 - 2.0 * u)
    for final_offset in offsets:
        lateral = final_offset * blend
        robot_xy = np.stack([lateral, forward], axis=-1).astype(np.float32)
        row = np.rint(robot_cell[0] - forward / CELL_SIZE_M).astype(np.int16)
        column = np.rint(
            robot_cell[1] + lateral / CELL_SIZE_M
        ).astype(np.int16)
        row = np.clip(row, 0, WORLD_GRID - 1)
        column = np.clip(column, 0, WORLD_GRID - 1)
        trajectories_robot.append(robot_xy)
        trajectories_cells.append(np.stack([row, column], axis=-1))
    return (
        np.stack(trajectories_robot),
        np.stack(trajectories_cells),
        offsets.astype(np.float32),
    )


def _trajectory_features(trajectory_cells, occupancy, terrain):
    """Accumulate task-reweightable geometry/risk features per trajectory."""
    features = np.zeros((N_TRAJECTORIES, 4), dtype=np.float32)
    segment_length = TRAJECTORY_HORIZON_M / (TRAJECTORY_SAMPLES - 1)
    for path_index, cells in enumerate(trajectory_cells):
        rows = cells[1:, 0]
        columns = cells[1:, 1]
        states = terrain[rows, columns]
        blocked = occupancy[rows, columns] == OCCUPIED
        features[path_index, 0] = segment_length * len(rows)
        features[path_index, 1] = segment_length * np.count_nonzero(
            (states == ROUGH) & ~blocked
        )
        features[path_index, 2] = segment_length * np.count_nonzero(
            (states == HAZARD) & ~blocked
        )
        features[path_index, 3] = segment_length * np.count_nonzero(blocked)
    return features


def task_costs(features, risks=(0.0, 0.55, 2.0, 25.0)):
    risks = np.asarray(risks, dtype=np.float32)
    # Base path length is always paid; the coefficients add task-dependent
    # penalties for normal, rough, hazardous and blocked segments.
    return (
        features[:, 0] * (1.0 + risks[0])
        + features[:, 1] * risks[1]
        + features[:, 2] * risks[2]
        + features[:, 3] * risks[3]
    ).astype(np.float32)


def _camera_rays():
    yy, xx = np.mgrid[0:IMAGE_SIDE, 0:IMAGE_SIDE].astype(np.float32)
    x = (xx + 0.5 - IMAGE_SIDE / 2.0) / (IMAGE_SIDE / 2.0)
    y = (IMAGE_SIDE / 2.0 - yy - 0.5) / (IMAGE_SIDE / 2.0)
    lateral = x * np.tan(np.deg2rad(HORIZONTAL_FOV_DEG / 2.0))
    forward = np.ones_like(lateral)
    vertical = y * np.tan(np.deg2rad(VERTICAL_FOV_DEG / 2.0))

    pitch = np.deg2rad(CAMERA_PITCH_DEG)
    pitched_forward = np.cos(pitch) * forward - np.sin(pitch) * vertical
    pitched_vertical = np.sin(pitch) * forward + np.cos(pitch) * vertical
    norm = np.sqrt(
        lateral * lateral
        + pitched_forward * pitched_forward
        + pitched_vertical * pitched_vertical
    )
    return lateral / norm, pitched_forward / norm, pitched_vertical / norm


RAY_LATERAL, RAY_FORWARD, RAY_VERTICAL = _camera_rays()


def _render_camera(occupancy, terrain, robot_cell, rng):
    """Vectorized 2.5-D ray caster returning aligned RGB-D-semantic data."""
    depth = np.full((IMAGE_SIDE, IMAGE_SIDE), MAX_DEPTH_M, dtype=np.float32)
    semantic = np.full((IMAGE_SIDE, IMAGE_SIDE), SEM_SKY, dtype=np.uint8)
    hit_row = np.full((IMAGE_SIDE, IMAGE_SIDE), -1, dtype=np.int16)
    hit_column = np.full((IMAGE_SIDE, IMAGE_SIDE), -1, dtype=np.int16)
    active = np.ones((IMAGE_SIDE, IMAGE_SIDE), dtype=bool)

    # A sub-cell marching step gives stable aligned RGB/depth/semantic hits.
    for distance in np.arange(0.12, MAX_DEPTH_M + 0.08, 0.08, dtype=np.float32):
        if not active.any():
            break
        world_z = CAMERA_HEIGHT_M + distance * RAY_VERTICAL
        world_row = np.floor(
            robot_cell[0] - distance * RAY_FORWARD / CELL_SIZE_M
        ).astype(np.int32)
        world_column = np.floor(
            robot_cell[1] + distance * RAY_LATERAL / CELL_SIZE_M
        ).astype(np.int32)
        inside = (
            (world_row >= 0)
            & (world_row < WORLD_GRID)
            & (world_column >= 0)
            & (world_column < WORLD_GRID)
        )
        safe_row = np.clip(world_row, 0, WORLD_GRID - 1)
        safe_column = np.clip(world_column, 0, WORLD_GRID - 1)
        obstacle_hit = (
            active
            & inside
            & (world_z >= 0.0)
            & (world_z <= OBSTACLE_HEIGHT_M)
            & (occupancy[safe_row, safe_column] == OCCUPIED)
        )
        ground_hit = active & inside & (world_z <= 0.0) & ~obstacle_hit
        escaped = active & ~inside

        if obstacle_hit.any():
            depth[obstacle_hit] = distance
            semantic[obstacle_hit] = SEM_OBSTACLE
            hit_row[obstacle_hit] = safe_row[obstacle_hit]
            hit_column[obstacle_hit] = safe_column[obstacle_hit]
        if ground_hit.any():
            depth[ground_hit] = distance
            ground_state = terrain[safe_row[ground_hit], safe_column[ground_hit]]
            ground_semantic = np.full(ground_state.shape, SEM_FLOOR, dtype=np.uint8)
            ground_semantic[ground_state == ROUGH] = SEM_ROUGH
            ground_semantic[ground_state == HAZARD] = SEM_HAZARD
            semantic[ground_hit] = ground_semantic
            hit_row[ground_hit] = safe_row[ground_hit]
            hit_column[ground_hit] = safe_column[ground_hit]
        active[obstacle_hit | ground_hit | escaped] = False

    rgb = SEMANTIC_COLORS[semantic].astype(np.float32)
    # Depth shading, simple floor texture and slight sensor noise make the
    # camera view visually first-person without encoding labels as flat tiles.
    visible = semantic != SEM_SKY
    fog = np.clip(1.08 - 0.035 * depth, 0.55, 1.0)
    rgb[visible] *= fog[visible, None]
    valid_cell = hit_row >= 0
    checker = ((hit_row + hit_column) % 2).astype(np.float32)
    floor = valid_cell & (semantic != SEM_OBSTACLE)
    rgb[floor] += (checker[floor, None] - 0.5) * 8.0
    sky_y = np.linspace(1.10, 0.82, IMAGE_SIDE, dtype=np.float32)[:, None]
    sky = semantic == SEM_SKY
    rgb[sky] *= np.broadcast_to(sky_y, sky.shape)[sky, None]
    noise = rng.normal(0.0, 2.2, size=rgb.shape).astype(np.float32)
    rgb = np.clip(rgb + noise, 0.0, 255.0).astype(np.uint8)
    return rgb, depth, semantic, hit_row, hit_column


def _region_correspondence(
    hit_row,
    hit_column,
    semantic,
    trajectory_cells,
    occupancy,
    terrain,
):
    """Map Qwen regions to visible BEV cells and trajectory cost features."""
    n_regions = SELECTOR_GRID**2
    patch_to_bev = np.full((n_regions, 2), -1, dtype=np.int16)
    patch_to_bev_valid = np.zeros(n_regions, dtype=np.float32)
    cell_region_votes = {}

    for patch_row in range(SELECTOR_GRID):
        for patch_column in range(SELECTOR_GRID):
            region = patch_row * SELECTOR_GRID + patch_column
            row_slice = slice(patch_row * PATCH_SIDE, (patch_row + 1) * PATCH_SIDE)
            col_slice = slice(
                patch_column * PATCH_SIDE, (patch_column + 1) * PATCH_SIDE
            )
            rows = hit_row[row_slice, col_slice]
            columns = hit_column[row_slice, col_slice]
            valid = rows >= 0
            patch_to_bev_valid[region] = float(valid.mean())
            if valid.any():
                patch_to_bev[region] = np.asarray(
                    [np.median(rows[valid]), np.median(columns[valid])],
                    dtype=np.int16,
                )
                cells, counts = np.unique(
                    np.stack([rows[valid], columns[valid]], axis=-1),
                    axis=0,
                    return_counts=True,
                )
                for cell, count in zip(cells, counts):
                    key = (int(cell[0]), int(cell[1]))
                    cell_region_votes.setdefault(key, []).append((region, int(count)))

    cell_to_region = {
        key: max(votes, key=lambda item: item[1])[0]
        for key, votes in cell_region_votes.items()
    }
    relation = np.zeros((N_TRAJECTORIES, n_regions), dtype=np.float32)
    region_features = np.zeros((N_TRAJECTORIES, n_regions, 4), dtype=np.float32)
    unmapped_features = np.zeros((N_TRAJECTORIES, 4), dtype=np.float32)
    segment_length = TRAJECTORY_HORIZON_M / (TRAJECTORY_SAMPLES - 1)

    for path_index, cells in enumerate(trajectory_cells):
        for row, column in cells[1:]:
            row = int(row)
            column = int(column)
            feature = np.zeros(4, dtype=np.float32)
            feature[0] = segment_length
            if occupancy[row, column] == OCCUPIED:
                feature[3] = segment_length
            elif terrain[row, column] == ROUGH:
                feature[1] = segment_length
            elif terrain[row, column] == HAZARD:
                feature[2] = segment_length

            # Search a one-cell neighborhood because a thin trajectory and a
            # finite-resolution camera ray do not always hit the exact cell.
            candidates = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    region = cell_to_region.get((row + dr, column + dc))
                    if region is not None:
                        candidates.append(region)
            if candidates:
                values, counts = np.unique(candidates, return_counts=True)
                region = int(values[np.argmax(counts)])
                relation[path_index, region] += 1.0
                region_features[path_index, region] += feature
            else:
                unmapped_features[path_index] += feature

    relation /= max(TRAJECTORY_SAMPLES - 1, 1)
    return (
        patch_to_bev,
        patch_to_bev_valid,
        relation,
        region_features,
        unmapped_features,
    )


def colorize_depth(depth):
    normalized = np.clip(depth / MAX_DEPTH_M, 0.0, 1.0)
    near = 1.0 - normalized
    rgb = np.stack(
        [255.0 * near, 255.0 * (1.0 - np.abs(2.0 * near - 1.0)), 255.0 * normalized],
        axis=-1,
    )
    return rgb.astype(np.uint8)


def colorize_semantic(semantic):
    return SEMANTIC_COLORS[semantic]


def render_bev(occupancy, terrain, robot_cell, scale=4):
    bev = np.empty((WORLD_GRID, WORLD_GRID, 3), dtype=np.uint8)
    bev[:] = (226, 228, 224)
    bev[terrain == ROUGH] = (185, 139, 69)
    bev[terrain == HAZARD] = (192, 62, 55)
    bev[occupancy == OCCUPIED] = (43, 48, 53)
    image = Image.fromarray(bev).resize(
        (WORLD_GRID * scale, WORLD_GRID * scale), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(image)
    center = ((int(robot_cell[1]) + 0.5) * scale, (int(robot_cell[0]) + 0.5) * scale)
    draw.ellipse(
        (center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4),
        fill=(35, 112, 210),
    )
    return np.asarray(image)


def build_one(rng, task_risks=(0.0, 0.55, 2.0, 25.0)):
    for _ in range(100):
        occupancy, terrain, robot_cell = _make_world(rng)
        trajectories_robot, trajectory_cells, offsets = _motion_primitives(robot_cell)
        features = _trajectory_features(trajectory_cells, occupancy, terrain)
        collision_free = features[:, 3] == 0.0
        if np.count_nonzero(collision_free) >= 2:
            break
    else:
        raise RuntimeError("failed to generate a world with two feasible trajectories")

    costs = task_costs(features, task_risks)
    optimal = int(np.argmin(costs))
    rgb, depth, semantic, hit_row, hit_column = _render_camera(
        occupancy, terrain, robot_cell, rng
    )
    (
        patch_to_bev,
        patch_to_bev_valid,
        path_patch_relation,
        trajectory_region_features,
        trajectory_unmapped_features,
    ) = _region_correspondence(
        hit_row,
        hit_column,
        semantic,
        trajectory_cells,
        occupancy,
        terrain,
    )
    path_patch_mask = (path_patch_relation > 0.0).astype(np.float32)
    local_bev_gt = np.zeros_like(occupancy, dtype=np.uint8)
    local_bev_gt[occupancy == OCCUPIED] = SEM_OBSTACLE
    local_bev_gt[(occupancy == FREE) & (terrain == NORMAL)] = SEM_FLOOR
    local_bev_gt[(occupancy == FREE) & (terrain == ROUGH)] = SEM_ROUGH
    local_bev_gt[(occupancy == FREE) & (terrain == HAZARD)] = SEM_HAZARD

    return {
        # Student input. ``image`` is a compatibility alias for existing Qwen
        # image preprocessing; both arrays contain the same first-person RGB.
        "rgb": rgb,
        "image": rgb,
        # Privileged perception targets; not default student inputs.
        "depth_gt": depth.astype(np.float32),
        "semantic_gt": semantic,
        "depth_vlm_image": colorize_depth(depth),
        "semantic_vlm_image": colorize_semantic(semantic),
        "local_bev_gt": local_bev_gt,
        "occupancy_gt": occupancy,
        "terrain_gt": terrain,
        "pixel_to_bev_row": hit_row,
        "pixel_to_bev_col": hit_column,
        "patch_to_bev": patch_to_bev,
        "patch_to_bev_valid": patch_to_bev_valid,
        # Canonical left-to-right local motion primitives and task labels.
        "trajectory_points_robot": trajectories_robot,
        "trajectory_cells": trajectory_cells,
        "trajectory_lateral_offsets": offsets,
        "trajectory_features": features,
        "trajectory_region_features": trajectory_region_features,
        "trajectory_unmapped_features": trajectory_unmapped_features,
        "candidate_path_patch_relation": path_patch_relation,
        "candidate_path_patch_mask": path_patch_mask,
        "trajectory_costs": costs,
        "optimal_path_idx": np.int64(optimal),
        "optimal_cost": np.float32(costs[optimal]),
        "task_risks": np.asarray(task_risks, dtype=np.float32),
        "task_text": np.asarray(DEFAULT_TASK_TEXT),
        # Camera metadata is required to define a camera image, but is not fed
        # into the student selector as a graph or pose feature.
        "robot_cell": robot_cell,
        "camera_height_m": np.float32(CAMERA_HEIGHT_M),
        "camera_pitch_deg": np.float32(CAMERA_PITCH_DEG),
        "horizontal_fov_deg": np.float32(HORIZONTAL_FOV_DEG),
        "vertical_fov_deg": np.float32(VERTICAL_FOV_DEG),
        "cell_size_m": np.float32(CELL_SIZE_M),
        "selector_grid": np.int64(SELECTOR_GRID),
        "student_input_mode": np.asarray("first_person_rgb_only"),
        "layout": np.asarray("first_person_rgb_local_trajectories_v1"),
    }


def render_preview(sample, output):
    rgb = Image.fromarray(sample["rgb"])
    depth = Image.fromarray(sample["depth_vlm_image"])
    segmentation = Image.fromarray(sample["semantic_vlm_image"])
    bev = Image.fromarray(
        render_bev(sample["occupancy_gt"], sample["terrain_gt"], sample["robot_cell"])
    )
    width = IMAGE_SIDE
    gap = 22
    top = 42
    canvas = Image.new("RGB", (4 * width + 3 * gap, width + 92), (246, 246, 244))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 14) if Path(font_path).exists() else None
    draw = ImageDraw.Draw(canvas)
    panels = [rgb, depth, segmentation, bev]
    titles = [
        "A. Student input: first-person RGB",
        "B. Depth GT (optional sensor/teacher)",
        "C. Semantic GT (teacher)",
        "D. Hidden BEV + trajectories",
    ]

    # Show the exact Qwen selector regions on the RGB input.
    rgb_draw = ImageDraw.Draw(panels[0])
    for index in range(SELECTOR_GRID + 1):
        coordinate = index * PATCH_SIDE
        rgb_draw.line((coordinate, 0, coordinate, width), fill=(235, 235, 235), width=1)
        rgb_draw.line((0, coordinate, width, coordinate), fill=(235, 235, 235), width=1)

    # Trajectories are debug-only and appear only in the hidden BEV panel.
    bev_draw = ImageDraw.Draw(panels[3])
    scale = width / WORLD_GRID
    for index, cells in enumerate(sample["trajectory_cells"]):
        points = [
            ((float(column) + 0.5) * scale, (float(row) + 0.5) * scale)
            for row, column in cells
        ]
        color = (42, 205, 89) if index == int(sample["optimal_path_idx"]) else (70, 112, 190)
        bev_draw.line(points, fill=color, width=4 if index == int(sample["optimal_path_idx"]) else 1)

    for index, (panel, title) in enumerate(zip(panels, titles)):
        x = index * (width + gap)
        canvas.paste(panel, (x, top))
        draw.text((x, 10), title, fill=(24, 24, 24), font=font)
    draw.text(
        (0, top + width + 20),
        "Only panel A is the default Qwen3-VL image. B/C/D are aligned supervision and debug data.",
        fill=(35, 35, 35),
        font=font,
    )
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data_v81")
    parser.add_argument("--split", choices=("train", "val", "test", "preview"), default="preview")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--preview", default="camera_v81_preview.png")
    parser.add_argument("--risk-normal", type=float, default=0.0)
    parser.add_argument("--risk-rough", type=float, default=0.55)
    parser.add_argument("--risk-hazard", type=float, default=2.0)
    parser.add_argument("--risk-blocked", type=float, default=25.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing .npz files in this exact split before generation.",
    )
    args = parser.parse_args()
    if args.n < 1:
        raise ValueError("--n must be positive")

    task_risks = (
        args.risk_normal,
        args.risk_rough,
        args.risk_hazard,
        args.risk_blocked,
    )
    master_rng = np.random.default_rng(args.seed)
    output_dir = Path(args.out) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.npz"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing)} .npz files. "
            "Use a new --out directory or pass --overwrite explicitly."
        )
    if args.overwrite:
        for old_file in existing:
            old_file.unlink()
    first = None
    for index in range(args.n):
        # Per-sample seeds keep the first N samples identical when a split is
        # later enlarged and make individual scenes reproducible.
        sample_seed = int(
            master_rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
        )
        sample = build_one(np.random.default_rng(sample_seed), task_risks)
        sample["split"] = np.asarray(args.split)
        sample["dataset_seed"] = np.int64(args.seed)
        sample["sample_seed"] = np.int64(sample_seed)
        sample["sample_index"] = np.int64(index)
        if first is None:
            first = sample
        np.savez_compressed(output_dir / f"{index:06d}.npz", **sample)
    render_preview(first, args.preview)
    print(f"saved {args.n} V8.1 samples to {output_dir}")
    print(f"saved preview: {args.preview}")
    print(
        f"RGB={first['rgb'].shape}, depth={first['depth_gt'].shape}, "
        f"semantic={first['semantic_gt'].shape}, "
        f"trajectories={len(first['trajectory_costs'])}, "
        f"optimal={int(first['optimal_path_idx'])}"
    )


if __name__ == "__main__":
    main()
