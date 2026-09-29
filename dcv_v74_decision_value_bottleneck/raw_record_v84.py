"""Write the small simulator-independent record consumed by V8.4 processing."""

from pathlib import Path

import numpy as np


def save_raw_record(
    output_path,
    image,
    reference_path_ego,
    traversable,
    dynamic_obstacle,
    map_bounds,
    goal_xy,
    ego_state,
    task_text,
    source_id,
    sample_id,
    expert_trajectory_ego=None,
):
    """Save arrays extracted inside either the nuPlan or Habitat environment.

    Coordinate convention:
      x points forward, y points left;
      map_bounds is [xmin, xmax, ymin, ymax];
      raster row 0 corresponds to ymax and column 0 corresponds to xmin.

    ``expert_trajectory_ego`` should be the nuPlan future ego trajectory or
    PointNav geodesic path.  If omitted, processing records an explicit
    reference-path fallback for smoke tests.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(
        image=np.asarray(image, dtype=np.uint8),
        reference_path_ego=np.asarray(reference_path_ego, dtype=np.float32),
        traversable=np.asarray(traversable, dtype=bool),
        dynamic_obstacle=np.asarray(dynamic_obstacle, dtype=bool),
        map_bounds=np.asarray(map_bounds, dtype=np.float32),
        goal_xy=np.asarray(goal_xy, dtype=np.float32),
        ego_state=np.asarray(ego_state, dtype=np.float32),
        task_text=np.str_(task_text),
        source_id=np.int64(source_id),
        sample_id=np.str_(sample_id),
    )
    if expert_trajectory_ego is not None:
        record["expert_trajectory_ego"] = np.asarray(
            expert_trajectory_ego, dtype=np.float32
        )
    np.savez_compressed(output_path, **record)
