"""Generate the sample pairs used by the web demo, and the demo video frames.

    python tools/make_samples.py

Writes
    docs/samples/<name>_before.png
    docs/samples/<name>_after.png
    docs/samples/<name>_truth.png     (ground truth, for the README figure)

The samples are deterministic: the same seeds always produce the same
pairs, so the figures in the README match what the demo shows.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.synth import generate_scene  # noqa: E402

# name -> (seed, forced event). The event is forced so each sample really
# shows the change type its filename claims -- a demo whose labels are
# wrong is worse than no demo.
SAMPLES = {
    "deforestation": (424242, ["deforestation"]),
    "urban_expansion": (515151, ["urban_expansion"]),
    "excavation": (616161, ["excavation"]),
    "flooding": (717171, ["flooding"]),
    "wildfire": (818181, ["wildfire_scar"]),
    "new_road": (919191, ["new_road"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="docs/samples")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    for name, (seed, events) in SAMPLES.items():
        s = generate_scene(seed, size=args.size, events=events)
        Image.fromarray(s["before"]).save(out / f"{name}_before.png")
        Image.fromarray(s["after"]).save(out / f"{name}_after.png")
        Image.fromarray(s["truth"]).save(out / f"{name}_truth.png")
        changed = (s["truth"] > 0).mean() * 100
        print(f"{name:16s} seed={seed:<8d} event={s['event']:<22s} changed={changed:5.2f}%")

    # the negative case: two acquisitions of unchanged ground
    s = generate_scene(313131, size=args.size, events=[])
    Image.fromarray(s["before"]).save(out / "clean_before.png")
    Image.fromarray(s["after"]).save(out / "clean_after.png")
    print(f"{'clean':16s} seed=313131   event={'none':<22s} changed={0.0:5.2f}%")

    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
