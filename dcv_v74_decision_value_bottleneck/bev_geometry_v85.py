"""Explicit BEV/legacy raster adapters; no camera or simulator dependencies."""

import numpy as np


RASTER_LAYOUT = "bev_x_rows_y_cols"
SCHEMA_REVISION = 3


def bev_to_legacy_raster(grid):
    """BEV: rows x descending, cols y descending -> legacy: rows y, cols x."""
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("metric raster must be two-dimensional")
    return np.ascontiguousarray(grid.T[:, ::-1])


def legacy_to_bev_raster(grid):
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError("metric raster must be two-dimensional")
    return np.ascontiguousarray(grid[:, ::-1].T)


def sample_bev_grid_nearest(grid, xy, bounds):
    """Sample BEV raster at ego-frame (x-forward, y-left) points."""
    from planning_data_v84 import sample_grid_nearest

    return sample_grid_nearest(bev_to_legacy_raster(grid), xy, bounds)
