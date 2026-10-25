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
    """Multi-octave smooth noise in [0, 1].

    The amplitude falls off by 0.5 per octave, which is the usual
    fractal-noise choice. For the *texture* octaves here that was too
    aggressive: the highest octave ended up contributing nearly as much
    high-frequency speckle as the base layer, and the result read as
    noise rather than as ground cover. 0.42 keeps the low frequencies
    dominant.
    """
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        s = 2 ** (octaves - o)
        small = rng.random((max(2, h // s), max(2, w // s))).astype(np.float32)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * up
        total += amp
        amp *= 0.42
    out /= max(total, 1e-6)
    lo, hi = out.min(), out.max()
    return (out - lo) / max(hi - lo, 1e-6)


def _tex_forest(rng, h, w):
    n = _fractal(h, w, rng, 6)
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([46, 74, 42], np.float32)              # lit canopy
    img += (n[..., None] - 0.5) * 74
    canopy = (rng.random((h, w)) > 0.975)[..., None] * np.array([46, 62, 26], np.float32)
    return np.clip(img + canopy, 0, 255)


def _tex_farm(rng, h, w):
    """Farmland: field strips with visible parcel boundaries."""
    n = _fractal(h, w, rng, 4)
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([124, 143, 72], np.float32)
    # strips, each with its own slight tone -- real farmland is a patchwork
    strip = np.floor(np.arange(w) / rng.integers(9, 16)).astype(np.int32)
    tone = rng.normal(0, 9, size=int(strip.max()) + 1)
    img += tone[strip][None, :, None]
    img += (n[..., None] - 0.5) * 30
    # parcel edges: a thin darker line every few strips
    edge = (strip.astype(np.int32) != np.roll(strip.astype(np.int32), 1))[None, :, None]
    img -= edge * np.array([22, 20, 12], np.float32)
    return np.clip(img, 0, 255)


def _tex_urban(rng, h, w):
    """Built-up area: irregular blocks, roads, and a little greenery.

    The buildings are made deliberately *lighter and greyer* than bare soil,
    with bright roof speckle. Without that separation the urban texture is
    just another brown, and the model cannot tell "new buildings" from
    "bare ground" — which is exactly what happened when the land cover
    changed to continuous noise fields (urban expansion dropped to IoU
    0.07). A real built-up area is recognisably grey against its
    surroundings.
    """
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([112, 111, 110], np.float32)          # grey, not brown
    # irregular building blocks rather than a uniform grid
    block = rng.random((h // 9 + 2, w // 9 + 2))
    up = cv2.resize(block.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    img += (up[..., None] - 0.5) * 40
    # bright roofs: a cue for "buildings", but sparse -- at 4.5% density
    # the area reads as a white speckle field rather than as a city.
    roofs = (rng.random((h, w)) > 0.988)[..., None] * np.array([44, 44, 42], np.float32)
    img += roofs
    # road network: darker than the roofs, and continuous
    roads = (np.arange(w) % 34 < 3)[None, :, None] | (np.arange(h) % 41 < 3)[:, None, None]
    img -= roads * np.array([30, 28, 26], np.float32)
    # scattered trees / open ground
    green = (rng.random((h, w)) > 0.972)[..., None] * np.array([-30, 4, -26], np.float32)
    return np.clip(img + green, 0, 255)


def _tex_water(rng, h, w):
    n = _fractal(h, w, rng, 3)
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([34, 68, 112], np.float32)             # lit water
    img += (n[..., None] - 0.5) * 22
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

    The horizontal ripple bands are what make the class *learnable* rather
    than merely distinguishable: they are a texture cue that survives the
    illumination and colour-cast differences between the two passes.
    """
    n = _fractal(h, w, rng, 4)
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([86, 84, 62], np.float32)          # sediment-laden
    img += (n[..., None] - 0.5) * 34
    ripple = np.sin(np.arange(h, dtype=np.float32)[:, None] / 3.5
                    + np.arange(w, dtype=np.float32)[None, :] / 9.0) * 11
    img += ripple[..., None]
    silt = (rng.random((h, w)) > 0.94)[..., None] * np.array([46, 38, 18], np.float32)
    return np.clip(img + silt, 0, 255)


def _tex_excavation(rng, h, w):
    """Open-pit excavation: terraced, freshly cut, much brighter than
    surrounding bare soil.

    The terraces are deliberately regular and horizontal. A real pit has
    benches cut at intervals, and that regularity is the cue the network
    can latch onto — an irregular blotch of brown would be
    indistinguishable from the bare soil it replaces.
    """
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([176, 148, 104], np.float32)         # fresh cut earth
    terrace = np.sin(np.arange(h, dtype=np.float32)[:, None] / 5.0) * 30
    img += terrace[..., None]
    # a sharp bench edge every ~15 rows
    bench = ((np.arange(h) % 15) < 2)[:, None, None] * np.array([34, 28, 18], np.float32)
    img += bench
    img += (rng.random((h, w, 1)) - 0.5) * 26
    return np.clip(img, 0, 255)


def _tex_newbuild(rng, h, w):
    """Fresh construction: bright concrete slabs and new roofs.

    Deliberably different from _tex_urban (the older built-up fabric):
    brighter, cleaner, with sharp-edged rectangles. A pair of images
    showing "more buildings" is only distinguishable if the new buildings
    look different from the old ones.
    """
    img = np.zeros((h, w, 3), np.float32)
    img += np.array([146, 145, 141], np.float32)          # fresh concrete
    # sharp-edged building blocks
    bs = rng.integers(6, 13)
    block = (rng.random((h // bs + 2, w // bs + 2)) > 0.42).astype(np.float32)
    up = cv2.resize(block, (w, h), interpolation=cv2.INTER_NEAREST)
    img += up[..., None] * np.array([22, 22, 20], np.float32)
    # bright new roofs
    roofs = (rng.random((h, w)) > 0.955)[..., None] * np.array([38, 38, 36], np.float32)
    img += roofs
    # bare ground between the buildings
    img -= (1 - up[..., None]) * np.array([14, 12, 8], np.float32)
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
    """A land-cover map with organic region boundaries.

    The first two versions built this from a label grid upscaled with
    nearest-neighbour, and the result read as pixel art rather than as an
    aerial image -- square blocks with staircased edges, which a reviewer
    noticed before the model did. Enlarging the grid and jittering it with
    a distance transform did not fix it either: the underlying grid is
    still a grid, and the boundaries stay axis-aligned.

    What works is generating the land cover from **continuous noise fields**
    and thresholding them. Each class owns a smooth scalar field; a pixel
    joins the class whose field is highest there. Boundaries then follow
    the noise contours, which are curved and irregular the way a treeline
    or a field edge actually is -- no grid anywhere in the process.
    """
    # three octaves: a broad "where is land vs water" field, a medium one
    # for the large classes, and a fine one to fray the edges.
    broad = _fractal(size, size, rng, 3)
    medium = _fractal(size, size, rng, 5)
    fine = _fractal(size, size, rng, 7)

    # water occupies the low end of the broad field, so it forms large
    # connected bodies rather than speckle
    water_field = broad - 0.15 * medium
    is_water = water_field < 0.42

    # The land classes need *large coherent regions*, not a salt-and-pepper
    # mix: farmland and forest cover tens of hectares each in a real scene.
    # The fine octave is therefore kept small -- just enough to fray the
    # boundary, not enough to shred the interior.
    fields = {
        _FOREST: 0.30 + 0.52 * medium + 0.06 * fine,
        _FARM: 0.34 + 0.50 * fine + 0.05 * medium,
        _URBAN: 0.32 + 0.46 * fine + 0.10 * medium,
        _BARE: 0.28 + 0.40 * medium + 0.06 * fine,
    }

    # Guarantee every land class appears. With independent noise fields the
    # ranking is close to random, and a scene can easily contain no urban
    # area at all -- which quietly removes the "new buildings" scenario from
    # the training mix. Each class is therefore given a random per-scene
    # bias, so over many scenes all of them turn up often.
    for c in fields:
        fields[c] = fields[c] + rng.uniform(-0.10, 0.10)

    out = np.zeros((size, size), np.int32)
    best = np.full((size, size), -1e9, np.float32)
    for c, f in fields.items():
        better = f > best
        best = np.where(better, f, best)
        out = np.where(better, c, out)

    # water wins wherever it is present
    out = np.where(is_water, _WATER, out)
    return out


def _render_land(rng, lm, size):
    out = np.zeros((size, size, 3), np.float32)
    for cls in np.unique(lm):
        m = lm == cls
        out[m] = _LAND_TEXTURES[int(cls)](rng, size, size)[m]
    # Desaturate towards a satellite-radiance look. Real imagery is not
    # postcard-green; pulling the chroma down by 25% and lifting the
    # blacks makes the scenes read as overhead photography rather than as
    # a map, and it also stops the change textures from being trivially
    # separable by colour alone.
    grey = out.mean(axis=2, keepdims=True)
    out = grey + (out - grey) * 0.75
    out = out * 0.93 + 14.0
    return out


def _blob(rng, size):
    """Soft-edged rotated ellipse as (mask, cy, cx).

    The edge is deliberately not a step function. A hard boundary would be
    unrealistically easy to segment (one pixel of contrast is all a local
    filter needs) and would flatter the model. A partially-mixed edge is
    what an actual footprint boundary looks like, and it is the case the
    model has to work for.
    """
    cy = rng.integers(0, size)
    cx = rng.integers(0, size)
    ry = rng.integers(int(size * 0.05), int(size * 0.15))
    rx = rng.integers(int(size * 0.05), int(size * 0.15))
    ang = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dy, dx = yy - cy, xx - cx
    u = dy * np.cos(ang) + dx * np.sin(ang)
    v = -dy * np.sin(ang) + dx * np.cos(ang)
    d = np.sqrt((u / max(ry, 1)) ** 2 + (v / max(rx, 1)) ** 2)
    # a soft ramp between 0.85 and 1.15 of the radius
    m = np.clip((1.15 - d) / 0.30, 0.0, 1.0)
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
                # NEW buildings, not more of the same: fresh concrete slabs
                # and bright roofs, distinct from the older built-up fabric
                # they are appearing next to. Without that distinction the
                # pair is ambiguous -- new concrete and old concrete look
                # the same in both frames.
                tex = _tex_newbuild(rng, size, size)
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
