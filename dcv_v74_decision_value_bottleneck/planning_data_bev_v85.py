"""Convert raw ego-centric BEV records into V8.5 planning samples."""

from pathlib import Path

import numpy as np

from planning_data_v84 import (
    MAX_CANDIDATES,
    MIN_CANDIDATES,
    process_raw_record,
)
from raw_record_bev_v85 import INPUT_MODE, STATIC_INPUT_MODE
from trajectory_gt_v85 import GT_CONTRACT, aligned_expert
from bev_geometry_v85 import (
    RASTER_LAYOUT, SCHEMA_REVISION, bev_to_legacy_raster, legacy_to_bev_raster,
)


SCHEMA_VERSION = 85


def _scalar(raw, name, default):
    value = raw.get(name)
    return default if value is None else value.item()


def process_bev_raw_record(
    raw,
    lateral_span=2.5,
    min_candidates=MIN_CANDIDATES,
    max_candidates=MAX_CANDIDATES,
):
    input_mode = str(_scalar(raw, "input_mode", ""))
    if input_mode == STATIC_INPUT_MODE:
        raise ValueError(
            "static_multitask records have task objectives, not logged route/GT targets. "
            "Do not use the legacy candidate/ADE processor or run_v85_rl.sh on these records. "
            "See README_V85_STATIC_BEV.md for the static export contract."
        )
    if input_mode != INPUT_MODE:
        raise ValueError(
            f"expected {INPUT_MODE!r}, got {input_mode!r}; "
            "front-camera V8.4 records are not accepted"
        )
    required = (
        "bev_rgb",
        "bev_semantic",
        "region_world_bounds",
        "region_semantic_counts",
        "bev_config_json",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise KeyError(f"BEV raw record is missing {missing}")

    compatibility = dict(raw)
    # Candidate-proxy preprocessing historically resampled GT by arc length
    # and rejected stationary logs. Its temporary target is replaced below.
    compatibility.pop("expert_trajectory_ego", None)
    compatibility["image"] = raw["bev_rgb"]
    # V8.4 metrics use x-columns; BEV uses x-rows. Adapt only the metric
    # rasters, not the visual image, route coordinates or world bounds.
    for name in ("traversable", "dynamic_obstacle"):
        compatibility[name] = bev_to_legacy_raster(raw[name])
    sample = process_raw_record(
        compatibility,
        lateral_span=lateral_span,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
    )
    expert, times, has_gt = aligned_expert(raw)
    sample.update(
        {
            "schema_version": np.int64(SCHEMA_VERSION),
            "schema_revision": np.int64(SCHEMA_REVISION),
            "gt_contract": np.str_(GT_CONTRACT),
            "expert_trajectory": expert[:, :2],
            "expert_heading": expert[:, 2],
            "planning_time_s": times,
            "trajectory_gt_available": np.bool_(has_gt),
            "expert_is_fallback": np.bool_(not has_gt),
            "traversable": np.asarray(raw["traversable"], dtype=bool),
            "dynamic_obstacle": np.asarray(raw["dynamic_obstacle"], dtype=bool),
            "metric_raster_layout": np.str_(RASTER_LAYOUT),
            "sdf": legacy_to_bev_raster(sample["sdf"]),
            "input_mode": np.str_(INPUT_MODE),
            "image": np.asarray(raw["bev_rgb"], dtype=np.uint8),
            "bev_rgb": np.asarray(raw["bev_rgb"], dtype=np.uint8),
            "bev_semantic": np.asarray(raw["bev_semantic"], dtype=np.uint8),
            "bev_semantic_names": np.asarray(
                raw["bev_semantic_names"], dtype="U32"
            ),
            "bev_config_json": np.str_(raw["bev_config_json"].item()),
            "region_world_bounds": np.asarray(
                raw["region_world_bounds"], dtype=np.float32
            ),
            "region_semantic_counts": np.asarray(
                raw["region_semantic_counts"], dtype=np.float32
            ),
            "scene_type": np.str_(_scalar(raw, "scene_type", "unknown")),
            "log_name": np.str_(_scalar(raw, "log_name", "")),
            "scenario_token": np.str_(
                _scalar(raw, "scenario_token", "")
            ),
            "timestamp_us": np.int64(_scalar(raw, "timestamp_us", 0)),
            "iteration": np.int64(_scalar(raw, "iteration", 0)),
        }
    )
    return sample


def convert_bev_directory(
    raw_root,
    output_root,
    min_candidates=MIN_CANDIDATES,
    max_candidates=MAX_CANDIDATES,
):
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    paths = sorted(raw_root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no raw BEV NPZ files found under {raw_root}")
    for raw_path in paths:
        with np.load(raw_path, allow_pickle=False) as data:
            raw = {key: data[key] for key in data.files}
        sample = process_bev_raw_record(
            raw,
            lateral_span=2.5,
            min_candidates=min_candidates,
            max_candidates=max_candidates,
        )
        output_path = output_root / raw_path.relative_to(raw_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **sample)
