"""Build the before/after/detected triptychs used in the README.

    python tools/make_triptychs.py

One PNG per change type: earlier acquisition, later acquisition, and the
model's own overlay, side by side with the measured numbers underneath.

These are the figures a reviewer sees without clicking anything, so the
numbers printed on them come from the model's own output rather than from
anywhere else.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import infer  # noqa: E402

SAMPLES = ROOT / "docs" / "samples"
FIGS = ROOT / "docs" / "figures"

CLASS_COLORS = {
    1: (232, 62, 44),
    2: (245, 138, 33),
    3: (214, 190, 60),
    4: (240, 226, 92),
    5: (52, 140, 226),
    6: (150, 96, 200),
}
CLASS_LABELS = {
    1: "Deforestation",
    2: "Urban expansion",
    3: "Excavation",
    4: "New road",
    5: "Flooding",
    6: "Wildfire scar",
}
ORIGIN = {c: ("human-made" if c <= 4 else "natural") for c in range(1, 7)}

ORDER = [
    ("deforestation", "human-made"),
    ("urban_expansion", "human-made"),
    ("excavation", "human-made"),
    ("new_road", "human-made"),
    ("flooding", "natural"),
    ("wildfire", "natural"),
]


def overlay(before, seg):
    arr = np.asarray(Image.fromarray(before).convert("RGB"), dtype=np.float32).copy()
    for c in range(1, 7):
        m = seg == c
        if m.any():
            arr[m] = 0.38 * arr[m] + 0.62 * np.array(CLASS_COLORS[c], np.float32)
    return arr.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/weights.npz")
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    weights = infer.load_weights(ROOT / args.weights)

    for name, origin in ORDER:
        before = np.asarray(Image.open(SAMPLES / f"{name}_before.png").convert("RGB"))
        after = np.asarray(Image.open(SAMPLES / f"{name}_after.png").convert("RGB"))
        seg, prob = infer.predict(before, after, weights)
        ov = overlay(before, seg)

        changed = seg > 0
        frac = changed.mean() * 100
        found = {c: int((seg == c).sum()) for c in range(1, 7) if (seg == c).any()}
        main_class = max(found, key=found.get) if found else None

        strip = np.concatenate([before, after, ov], axis=1)
        img = Image.fromarray(strip)
        # scale down to a comfortable width for the README
        w = 900
        img = img.resize((w, int(img.height * w / img.width)), Image.LANCZOS)
        img.save(FIGS / f"case_{name}.png")

        label = CLASS_LABELS.get(main_class, "no change")
        print(
            f"{name:16s} -> {label:16s} ({origin:10s}) "
            f"changed {frac:5.2f}%  {found}"
        )

    print(f"\nsaved -> {FIGS}")


if __name__ == "__main__":
    main()
