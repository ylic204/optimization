import numpy as np
from PIL import Image, ImageDraw


STATE_NAMES = {
    0: "normal",
    1: "rough",
    2: "hazard",
    3: "blocked",
}


BASE = (124, 124, 124)
DARK = (88, 91, 92)


def _base(tile, rng):
    arr = np.full(
        (tile, tile, 3),
        124,
        dtype=np.int16,
    )

    # Low-amplitude common road texture.
    noise = rng.integers(
        -3,
        4,
        size=(tile, tile, 1),
    )

    arr += noise

    return Image.fromarray(
        np.clip(
            arr,
            0,
            255,
        ).astype(
            np.uint8
        )
    )


def _normal_tile(tile, rng):
    return _base(
        tile,
        rng,
    )


def _rough_tile(tile, rng):
    img = _base(
        tile,
        rng,
    )

    d = ImageDraw.Draw(
        img
    )

    # Distributed gravel / small stones.
    # Total anomalous area is comparable to hazard/blocked, but spatial shape
    # is clearly different at 32x32 resolution.
    for _ in range(18):
        x = int(
            rng.integers(
                3,
                tile - 3,
            )
        )

        y = int(
            rng.integers(
                3,
                tile - 3,
            )
        )

        r = int(
            rng.integers(
                1,
                3,
            )
        )

        d.ellipse(
            (
                x-r,
                y-r,
                x+r,
                y+r,
            ),
            fill=DARK,
        )

    return img


def _hazard_tile(tile, rng):
    img = _base(
        tile,
        rng,
    )

    d = ImageDraw.Draw(
        img
    )

    # Compact wet/slippery region. Use only a subtle cool tint; the main
    # distinction from rough terrain is spatial organization, not gross color.
    cx = int(
        rng.integers(
            11,
            tile - 10,
        )
    )

    cy = int(
        rng.integers(
            11,
            tile - 10,
        )
    )

    rx = int(
        rng.integers(
            6,
            9,
        )
    )

    ry = int(
        rng.integers(
            4,
            7,
        )
    )

    d.ellipse(
        (
            cx-rx,
            cy-ry,
            cx+rx,
            cy+ry,
        ),
        fill=(
            86,
            96,
            101,
        ),
    )

    # Specular micro-highlights make the full-resolution crop look like a
    # puddle rather than a generic dark blob.
    for _ in range(5):
        x = int(
            rng.integers(
                max(
                    cx-rx+2,
                    1,
                ),
                min(
                    cx+rx-1,
                    tile-1,
                ),
            )
        )

        y = int(
            rng.integers(
                max(
                    cy-ry+1,
                    1,
                ),
                min(
                    cy+ry,
                    tile-1,
                ),
            )
        )

        d.point(
            (
                x,
                y,
            ),
            fill=(
                150,
                154,
                155,
            ),
        )

    return img


def _blocked_tile(tile, rng):
    img = _base(
        tile,
        rng,
    )

    d = ImageDraw.Draw(
        img
    )

    # Thin continuous physical barrier / fallen object.
    # Random orientation prevents the 3x3 preview from trivially using one
    # fixed line location as a class code.
    horizontal = (
        rng.random()
        < 0.5
    )

    if horizontal:
        y = int(
            rng.integers(
                8,
                tile - 8,
            )
        )

        d.line(
            (
                4,
                y,
                tile - 5,
                y,
            ),
            fill=DARK,
            width=3,
        )

        d.line(
            (
                7,
                y - 3,
                7,
                y + 3,
            ),
            fill=DARK,
            width=2,
        )

        d.line(
            (
                tile - 8,
                y - 3,
                tile - 8,
                y + 3,
            ),
            fill=DARK,
            width=2,
        )

    else:
        x = int(
            rng.integers(
                8,
                tile - 8,
            )
        )

        d.line(
            (
                x,
                4,
                x,
                tile - 5,
            ),
            fill=DARK,
            width=3,
        )

        d.line(
            (
                x - 3,
                7,
                x + 3,
                7,
            ),
            fill=DARK,
            width=2,
        )

        d.line(
            (
                x - 3,
                tile - 8,
                x + 3,
                tile - 8,
            ),
            fill=DARK,
            width=2,
        )

    return img


def render_tile(
    state,
    tile,
    rng,
):
    if state == 0:
        return _normal_tile(
            tile,
            rng,
        )

    if state == 1:
        return _rough_tile(
            tile,
            rng,
        )

    if state == 2:
        return _hazard_tile(
            tile,
            rng,
        )

    if state == 3:
        return _blocked_tile(
            tile,
            rng,
        )

    raise ValueError(
        state
    )


def render_scene(
    states_by_patch,
    cfg,
    rng,
):
    """
    One 192x192 high-resolution raw scene.

    At high resolution:
      rough   -> distributed gravel;
      hazard  -> compact puddle;
      blocked -> continuous barrier.

    At preview resolution (18x18 = 3x3 per scene cell):
      all three mainly become a coarse "abnormal" appearance, while exact
      severity/geometry is intentionally ambiguous.
    """
    canvas = Image.new(
        "RGB",
        (
            cfg.image_size,
            cfg.image_size,
        ),
        BASE,
    )

    for j in range(
        cfg.n_patches
    ):
        state = int(
            states_by_patch[
                j
            ]
        )

        tile_img = render_tile(
            state,
            cfg.tile,
            rng,
        )

        r, c = divmod(
            j,
            cfg.grid,
        )

        canvas.paste(
            tile_img,
            (
                c
                * cfg.tile,
                r
                * cfg.tile,
            ),
        )

    return np.asarray(
        canvas,
        dtype=np.uint8,
    )
