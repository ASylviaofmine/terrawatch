"""Synthetic satellite-scene generator.

Produces triplets: a procedurally rendered aerial scene, a second acquisition
of the same ground with a known change event applied, and a pixel-wise class
map of that event.

Why synthetic: the problem statement explicitly allows simulated
environments, and a synthetic source gives *exact* ground truth -- which is
what turns the reported precision/recall/IoU into measurements instead of
decoration.

Class-map values
    0  no change
    1  deforestation      (human-made)
    2  urban expansion    (human-made)
    3  excavation         (human-made)
    4  new road           (human-made)
    5  flooding           (natural)
    6  wildfire scar      (natural)

Only numpy + OpenCV are needed; this module is offline tooling (dataset
generation), not part of the deployed web app.
"""

import cv2
import numpy as np

CLASSES = [
    "no_change",
    "deforestation",
    "urban_expansion",
    "excavation",
    "new_road",
    "flooding",
    "wildfire_scar",
]

HUMAN_CLASSES = {1, 2, 3, 4}
NATURAL_CLASSES = {5, 6}

# land-cover labels used to build the *before* scene
_FOREST, _FARM, _URBAN, _WATER, _BARE = 0, 1, 2, 3, 4

_EVENT_CLASS = {
    "deforestation": 1,
    "urban_expansion": 2,
    "excavation": 3,
    "new_road": 4,
    "flooding": 5,
    "wildfire_scar": 6,
}

_EVENT_SURFACE = {
    "deforestation": _FOREST,
    "urban_expansion": _URBAN,
    "excavation": _BARE,
    "new_road": None,
    # A flood does not need a particular land cover underneath -- it can
    # inundate farmland, grassland or the edge of a settlement. Leaving this
    # as None means the blob lands wherever it lands.
    "flooding": None,
    "wildfire_scar": _FOREST,
}

# How often each event occurs in the training mix. Flooding and excavation
# were raised after the first evaluation showed them at IoU 0.03 and 0.00 --
# not because they are rare in reality, but because the synthetic textures
# were too easily confused with the scene's existing bare soil and water.
_EVENT_WEIGHT = {
    "deforestation": 0.24,
    "urban_expansion": 0.17,
    "excavation": 0.19,
    "new_road": 0.13,
    "flooding": 0.14,
    "wildfire_scar": 0.13,
}


# ---------------------------------------------------------------------------
# texture synthesis
# ---------------------------------------------------------------------------

def _fractal(h, w, rng, octaves=5):
    """Multi-octave smooth noise in [0, 1]."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        s = 2 ** (octaves - o)
        small = rng.random((max(2, h // s), max(2, w // s))).astype(np.float32)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * up
        total += amp
        amp *= 0.55
    out /= max(total, 1e-6)
    lo, hi = out.min(), out.max()
    return (out - lo) / max(hi - lo, 1e-6)


def _tex_forest(rng, h, w):
    n = _fractal(h, w, rng, 6)
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([26, 58, 30], np.float32)
    img += (n[..., None] - 0.5) * 74
    canopy = (rng.random((h, w)) > 0.982)[..., None] * np.array([46, 62, 26], np.float32)
    return np.clip(img + canopy, 0, 255)


def _tex_farm(rng, h, w):
    n = _fractal(h, w, rng, 4)
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([124, 143, 72], np.float32)
    stripe = np.sin(np.arange(w, dtype=np.float32)[None, :] / 7.0 + rng.uniform(0, 6.0))
    img += (stripe[..., None] * 16) + (n[..., None] - 0.5) * 34
    return np.clip(img, 0, 255)


def _tex_urban(rng, h, w):
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([104, 100, 97], np.float32)
    block = rng.random((h // 12 + 2, w // 12 + 2))
    up = cv2.resize(block.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    img += (up[..., None] - 0.5) * 58
    roads = (np.arange(w) % 34 < 3)[None, :, None] | (np.arange(h) % 41 < 3)[:, None, None]
    img += roads * np.array([38, 36, 33], np.float32)
    return np.clip(img, 0, 255)


def _tex_water(rng, h, w):
    n = _fractal(h, w, rng, 3)
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([20, 52, 104], np.float32)
    img += (n[..., None] - 0.5) * 20
    return np.clip(img, 0, 255)


def _tex_bare(rng, h, w):
    n = _fractal(h, w, rng, 5)
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([146, 116, 79], np.float32)
    img += (n[..., None] - 0.5) * 52
    return np.clip(img, 0, 255)


def _tex_road(rng, h, w):
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([168, 162, 152], np.float32)
    img += (rng.random((h, w, 1)) - 0.5) * 22
    return img


def _tex_stump(rng, h, w):
    """Deforested ground: bare soil with residual canopy specks."""
    img = _tex_bare(rng, h, w)
    stumps = (rng.random((h, w)) > 0.985)[..., None] * np.array([58, 46, 26], np.float32)
    return np.clip(img + stumps, 0, 255)


def _tex_char(rng, h, w):
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([44, 34, 30], np.float32)
    img += (rng.random((h, w, 1)) - 0.5) * 30
    ember = (rng.random((h, w)) > 0.994)[..., None] * np.array([70, 28, 8], np.float32)
    return np.clip(img + ember, 0, 255)


def _tex_flood(rng, h, w):
    """Flood water: muddied, sediment-laden — NOT the same as clear lake water.

    This distinction matters. If the flood texture were identical to the
    scene's existing water, the pair would be genuinely ambiguous: a pixel
    that is blue in both frames is a lake in one and a flood in the other,
    and no model could resolve it from the two images alone. Real floods
    are turbid and brown, so the flood texture is deliberately muddier and
    warmer than the standing water it covers.
    """
    n = _fractal(h, w, rng, 4)
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([86, 84, 62], np.float32)          # sediment-laden
    img += (n[..., None] - 0.5) * 34
    silt = (rng.random((h, w)) > 0.94)[..., None] * np.array([46, 38, 18], np.float32)
    return np.clip(img + silt, 0, 255)


def _tex_excavation(rng, h, w):
    """Open-pit excavation: terraced, freshly cut, much brighter than
    surrounding bare soil."""
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([176, 148, 104], np.float32)         # fresh cut earth
    terrace = np.sin(np.arange(h, dtype=np.float32)[:, None] / 5.0) * 30
    img += terrace[..., None]
    img += (rng.random((h, w, 1)) - 0.5) * 26
    return np.clip(img, 0, 255)


_LAND_TEXTURES = {
    _FOREST: _tex_forest,
    _FARM: _tex_farm,
    _URBAN: _tex_urban,
    _WATER: _tex_water,
    _BARE: _tex_bare,
}


# ---------------------------------------------------------------------------
# scene construction
# ---------------------------------------------------------------------------

def _land_cover(rng, size):
    low = 14
    lm = rng.integers(0, 5, size=(low, low)).astype(np.int32)
    # push water towards one side so the scene is not uniform
    lm[:, : low // 3] = _WATER
    return cv2.resize(lm, (size, size), interpolation=cv2.INTER_NEAREST)


def _render_land(rng, lm, size):
    out = np.zeros((size, size, 3), np.float32)
    for cls in np.unique(lm):
        m = lm == cls
        out[m] = _LAND_TEXTURES[int(cls)](rng, size, size)[m]
    return out


def _blob(rng, size):
    """Soft-edged rotated ellipse as (mask, cy, cx)."""
    cy = rng.integers(0, size)
    cx = rng.integers(0, size)
    ry = rng.integers(int(size * 0.05), int(size * 0.15))
    rx = rng.integers(int(size * 0.05), int(size * 0.15))
    ang = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dy, dx = yy - cy, xx - cx
    u = dy * np.cos(ang) + dx * np.sin(ang)
    v = -dy * np.sin(ang) + dx * np.cos(ang)
    d = (u / max(ry, 1)) ** 2 + (v / max(rx, 1)) ** 2
    m = (d < 1.0).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 1.6)
    return m, cy, cx


def _pick_center(rng, lm, target):
    ys, xs = np.nonzero(lm == target)
    if len(ys) == 0:
        return None
    i = rng.integers(len(ys))
    return int(ys[i]), int(xs[i])


def _road_polyline(rng, size):
    n = rng.integers(3, 6)
    pts = [(rng.integers(0, size), rng.integers(0, size)) for _ in range(n)]
    return np.array(pts, np.int32)


def generate_scene(seed, size=256, events=None):
    """Return a dict with before/after RGB uint8 and the truth class map.

    Args:
        seed:    RNG seed; the scene is fully determined by it.
        size:    image side length in pixels.
        events:  optional list of event names to force, e.g. ["flooding"].
                 When omitted the event mix is sampled from the weights.
    """
    rng = np.random.default_rng(seed)
    lm = _land_cover(rng, size)
    before = _render_land(rng, lm, size)

    after = before.copy()
    truth = np.zeros((size, size), np.uint8)

    names = list(_EVENT_WEIGHT.keys())
    if events is None:
        probs = np.array([_EVENT_WEIGHT[n] for n in names], np.float64)
        probs /= probs.sum()
        n_events = 1 if rng.random() < 0.72 else 2
        chosen = rng.choice(len(names), size=n_events, replace=False, p=probs)
        chosen_names = [names[c] for c in chosen]
    else:
        chosen_names = list(events)

    applied = []
    for name in chosen_names:
        cls_id = _EVENT_CLASS[name]
        surface = _EVENT_SURFACE[name]

        if name == "new_road":
            pts = _road_polyline(rng, size)
            canvas = np.zeros((size, size), np.uint8)
            cv2.polylines(canvas, [pts], False, 255, thickness=rng.integers(8, 13))
            m = canvas.astype(np.float32) / 255.0
            # a new road is new *asphalt over bare ground*: the "before" side
            # of the corridor must not already look like a road, otherwise the
            # pair shows no change at all. So the corridor is forced to bare
            # soil in the before image.
            tex = _tex_road(rng, size, size)
            corridor = np.zeros((size, size), np.uint8)
            cv2.polylines(corridor, [pts], False, 255, thickness=rng.integers(8, 13))
            cm = corridor.astype(np.float32) / 255.0
            before = before * (1 - cm[..., None]) + _tex_bare(rng, size, size) * cm[..., None]
        else:
            m, _, _ = _blob(rng, size)
            if surface is not None:
                centre = _pick_center(rng, lm, surface)
                if centre is None:
                    continue
                # re-centre the blob on a pixel of the required land cover
                ys, xs = np.nonzero(m > 0.5)
                if len(ys) == 0:
                    continue
                shift_y = centre[0] - int(ys.mean())
                shift_x = centre[1] - int(xs.mean())
                m = np.roll(np.roll(m, shift_y, axis=0), shift_x, axis=1)
            if name == "deforestation":
                tex = _tex_stump(rng, size, size)
            elif name == "urban_expansion":
                tex = _tex_urban(rng, size, size)
            elif name == "excavation":
                tex = _tex_excavation(rng, size, size)
            elif name == "flooding":
                tex = _tex_flood(rng, size, size)
            else:
                tex = _tex_char(rng, size, size)

        after = after * (1 - m[..., None]) + tex * m[..., None]
        truth = np.where(m > 0.5, cls_id, truth)
        applied.append(name)

    # acquisition differences: a different pass has different illumination
    before = np.clip(before * (1.0 + rng.normal(0, 0.035)), 0, 255)
    after = np.clip(after * (1.0 + rng.normal(0, 0.035)), 0, 255)
    after = np.clip(
        after + np.array([rng.normal(0, 4), rng.normal(0, 2), rng.normal(0, 4)], np.float32),
        0,
        255,
    )
    after[..., 2] -= 8.0  # the second pass is slightly warmer / less blue
    before += rng.normal(0, 3.2, before.shape)
    after += rng.normal(0, 3.2, after.shape)

    return {
        "before": np.clip(before, 0, 255).astype(np.uint8),
        "after": np.clip(after, 0, 255).astype(np.uint8),
        "truth": truth.astype(np.uint8),
        "event": "+".join(applied) if applied else "none",
    }
