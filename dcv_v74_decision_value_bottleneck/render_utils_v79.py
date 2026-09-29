"""Rendering helpers for the visible V7.9 spatial corridor."""

import numpy as np
from PIL import Image, ImageDraw

from render_utils_v7 import render_scene
from spatial_paths_v79 import patch_coordinate


def _patch_center(index, cfg):
    row, column = patch_coordinate(index, cfg.grid)
    half = cfg.tile // 2
    return column * cfg.tile + half, row * cfg.tile + half


def render_spatial_corridor(patch_state, topology, cfg, rng):
    """Render terrain plus an observable graph whose edges occupy patches."""
    image = Image.fromarray(render_scene(patch_state, cfg, rng))
    draw = ImageDraw.Draw(image)
    hubs = topology["hub_patches"]
    edge_patch = topology["edge_patch"]

    # Draw the candidate route geometry.  Every two-segment polyline passes
    # through the center of exactly one edge patch.
    for edge_index, (source, target) in enumerate(topology["edges"]):
        points = [
            _patch_center(hubs[int(source)], cfg),
            _patch_center(edge_patch[edge_index], cfg),
            _patch_center(hubs[int(target)], cfg),
        ]
        draw.line(points, fill=(205, 205, 205), width=max(2, cfg.tile // 12))

    # A thin frame makes the edge-to-patch correspondence visually explicit.
    for index in edge_patch:
        row, column = patch_coordinate(index, cfg.grid)
        left = column * cfg.tile
        top = row * cfg.tile
        draw.rectangle(
            (left + 1, top + 1, left + cfg.tile - 2, top + cfg.tile - 2),
            outline=(178, 178, 178),
            width=1,
        )

    radius = max(4, cfg.tile // 6)
    for hub_index, patch in enumerate(hubs):
        x, y = _patch_center(patch, cfg)
        if hub_index == 0:
            fill = (92, 128, 92)
        elif hub_index == len(hubs) - 1:
            fill = (128, 92, 92)
        else:
            fill = (105, 105, 105)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
        if hub_index in (0, len(hubs) - 1):
            label = "S" if hub_index == 0 else "G"
            draw.text((x - 3, y - 5), label, fill=(235, 235, 235))

    return np.asarray(image, dtype=np.uint8)
