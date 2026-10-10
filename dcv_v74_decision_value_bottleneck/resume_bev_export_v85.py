"""Atomic static-frame writes and validated resumption, including older r7 frames."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

PARAMETERS = (
    "map_version", "planning_mode", "tasks", "goal_source", "gt_horizon_s", "gt_sample_interval_s",
    "skip_numeric_reference", "goal_xy", "goal_min_distance", "goal_max_distance", "goal_margin_m",
    "goal_seed", "goal_min_forward_m", "station_step_m", "lateral_offsets_m", "collision_step_m",
    "min_turn_radius_m", "safety_margin_m", "lane_change_policy", "hazard_buffer_m",
    "pedestrian_weight", "pedestrian_scale_m", "nominal_speed", "save_preview",
    "audit_near_distance_m", "audit_multi_vehicle_min",
)
MARKER = "export_complete.json"


def _digest(path):
    digest = hashlib.sha256()
    if path.suffix == ".npz":
        # ZIP timestamps/compression are not part of the data contract.
        with np.load(path, allow_pickle=False) as record:
            for key in sorted(record.files):
                array = np.ascontiguousarray(record[key])
                digest.update(json.dumps([key, array.dtype.str, array.shape]).encode())
                digest.update(array.tobytes())
    else:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _files(folder):
    return {str(path.relative_to(folder)): _digest(path)
            for path in sorted(folder.rglob("*")) if path.is_file() and path.name != MARKER}


def _verify(folder, expected):
    for relative, digest in expected.items():
        path = folder / relative
        if not path.is_file() or _digest(path) != digest:
            raise ValueError(f"Cannot resume incomplete/incompatible frame: {path}. "
                             "Existing results were preserved; use a new --output-root.")


def _save_marker(folder, protocol, files):
    marker = folder / MARKER
    temporary = folder / (MARKER+".tmp")
    temporary.write_text(json.dumps(dict(protocol=protocol, files=files), sort_keys=True, indent=2))
    temporary.replace(marker)


def export_static_frame(scenario, iteration, output_root, config, args, export_fn):
    relative = Path(scenario.log_name) / scenario.token / f"{iteration:04d}"
    root = Path(output_root)
    folder = root / relative
    protocol = dict(version="static_r7_evidence_audit_v1", config=json.loads(config.to_json()),
                    parameters={key: getattr(args, key) for key in PARAMETERS},
                    frame=str(relative), timestamp_us=int(scenario.get_time_point(iteration).time_us))
    marker = folder / MARKER
    if folder.exists():
        if not args.resume:
            raise FileExistsError(f"Frame already exists: {folder}. "
                                  "Use --resume to validate/reuse it, or choose a new --output-root.")
        if marker.is_file():
            saved = json.loads(marker.read_text())
            if saved["protocol"] != protocol:
                raise ValueError(f"Cannot resume with different export parameters: {folder}; use a new --output-root.")
            _verify(folder, saved["files"])
            return None
    root.mkdir(parents=True, exist_ok=True)
    # Publish only complete frames. Failed exports leave no partial final directory.
    with tempfile.TemporaryDirectory(prefix=".bev-export-", dir=root) as temporary:
        if not export_fn(scenario, iteration, temporary, config, args):
            if folder.exists():
                raise ValueError(f"Existing frame cannot be reproduced: {folder}; use a new --output-root.")
            return False
        generated = Path(temporary) / relative
        files = _files(generated)
        if folder.exists():
            # Pre-fix r7 has no completion marker: recompute once and compare all
            # outputs instead of assuming matching parameters or completeness.
            _verify(folder, files)
            _save_marker(folder, protocol, files)
            return None
        _save_marker(generated, protocol, files)
        folder.parent.mkdir(parents=True, exist_ok=True)
        generated.rename(folder)
    return True
