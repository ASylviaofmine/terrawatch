"""Pre-generate a dataset of synthetic scenes to disk.

    python tools/gen_dataset.py --n 400 --out data/train

Why: training generates scenes on the fly, which is elegant but makes the
CPU the bottleneck (the GPU sat at 5% utilisation). Pre-generating lets the
GPU do the work it is good at, and the cache can be regenerated with one
command.

Layout per split
    data/<split>/<i>_before.png
    data/<split>/<i>_after.png
    data/<split>/<i>_truth.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.synth import generate_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--out", default="data/train")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        s = generate_scene(args.seed + i, size=args.size)
        Image.fromarray(s["before"]).save(out / f"{i:06d}_before.png")
        Image.fromarray(s["after"]).save(out / f"{i:06d}_after.png")
        Image.fromarray(s["truth"]).save(out / f"{i:06d}_truth.png")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n}", flush=True)

    print(f"wrote {args.n} scenes -> {out}")


if __name__ == "__main__":
    main()
