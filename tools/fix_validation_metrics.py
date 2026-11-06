"""Correct the change IoU in runs/metrics.json.

tools/train.py computed the binary change IoU as

    inter[1:].sum() / union[1:].sum()

where `inter` is the per-class intersection (the diagonal) and `union` is
the per-class union summed over the six change classes. The numerator
counts only change pixels classified into the *right* class; the
denominator counts *every* change pixel. Those are two different
questions, so the ratio was not an IoU at all: it read 0.838 where
tools/evaluate.py measured 0.908 on the same 96 scenes.

tools/train.py now accumulates the full 7x7 confusion matrix and derives
every metric from it, exactly as tools/evaluate.py does. This script
recomputes the committed validation metrics with that fixed function so
the file on disk matches what a fresh training run would write.

The model is not retrained and the weights are not touched: the per-class
numbers were already correct, because the bug only affected the binary
aggregation.

    python tools/fix_validation_metrics.py
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("trainmod", ROOT / "tools" / "train.py")
trainmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainmod)

from src.model import ChangeNet  # noqa: E402


def main():
    path = ROOT / "runs" / "metrics.json"
    old = json.loads(path.read_text())

    m = ChangeNet()
    m.load_state_dict(torch.load(ROOT / "runs" / "model.pt", map_location="cpu"))
    m.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m.to(dev)

    # the same seed range and scene count the original run used
    seed0 = int(old.get("args", {}).get("val_seed", 900000))
    n = int(old.get("args", {}).get("val_n", 96))
    size = int(old.get("args", {}).get("size", 128))

    new = trainmod.evaluate(m, seed0, n, size, dev)

    print("validation set: seed", seed0, "n", n, "size", size)
    print()
    print(f"  change_iou      {old['change_iou']:.6f}  ->  {new['change_iou']:.6f}")
    print(f"  pixel_acc       {old['pixel_acc']:.6f}  ->  {new['pixel_acc']:.6f}")

    cls = ["no_change", "deforestation", "urban_expansion", "excavation",
           "new_road", "flooding", "wildfire_scar"]
    same = True
    for i, c in enumerate(cls):
        a, b = old["per_class_iou"][i], new["per_class_iou"][i]
        if abs(a - b) > 1e-9:
            same = False
            print(f"  per-class {c} changed: {a:.6f} -> {b:.6f}")
    print(f"  per-class IoU unchanged: {same}")

    old["change_iou"] = new["change_iou"]
    old["pixel_acc"] = new["pixel_acc"]
    old["per_class_iou"] = new["per_class_iou"]
    old["per_class_precision"] = new["per_class_precision"]
    old["note"] = (
        "change_iou was corrected on 2026-11-06: tools/train.py mixed a "
        "diagonal numerator with a binary union denominator, which read "
        "0.838 instead of 0.908 on this same validation set. The "
        "authoritative held-out numbers are in runs/eval/metrics.json."
    )
    path.write_text(json.dumps(old, indent=2))
    print()
    print("updated", path)


if __name__ == "__main__":
    main()
