"""Convert raw ego-centric BEV records into V8.5 planning samples."""

from pathlib import Path

import numpy as np

from planning_data_v84 import (
    MAX_CANDIDATES,
    MIN_CANDIDATES,
    process_raw_record,
)
from raw_record_bev_v85 import INPUT_MODE


SCHEMA_VERSION = 85
SCHEMA_REVISION = 1


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
    compatibility["image"] = raw["bev_rgb"]
    sample = process_raw_record(
        compatibility,
        lateral_span=lateral_span,
        min_candidates=min_candidates,
        max_candidates=max_candidates,
    )
    sample.update(
        {
            "schema_version": np.int64(SCHEMA_VERSION),
            "schema_revision": np.int64(SCHEMA_REVISION),
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
