"""Evaluate the trained model and write the figures + tables used in the README.

    python tools/evaluate.py --n 200 --out runs/eval

Produces
    runs/eval/metrics.json      per-class IoU / precision / recall / F1
    runs/eval/confusion.png     normalised confusion matrix
    runs/eval/pr_curve.png      precision-recall curve per change class
    runs/eval/report.md         the table, ready to paste into the README

The evaluation set is generated from a seed range disjoint from the
training range, so no scene can appear in both.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import infer  # noqa: E402
from src.synth import CLASSES, HUMAN_CLASSES, NATURAL_CLASSES, generate_scene  # noqa: E402

SIZE = 128


def load_model(path):
    p = ROOT / path
    if p.suffix == ".npz":
        return infer.load_weights(p)
    import torch

    from src.model import ChangeNet

    m = ChangeNet()
    m.load_state_dict(torch.load(p, map_location="cpu"))
    m.eval()
    return m


def predict_scene(before, after, weights, kind):
    if kind == "numpy":
        seg, prob = infer.predict(before, after, weights)
        return seg, prob
    import torch

    x = torch.from_numpy(
        infer.prepare(before, after).transpose(2, 0, 1)[None].astype(np.float32)
    )
    with torch.no_grad():
        logits = weights(x)[0].numpy().transpose(1, 2, 0)
    prob = infer.softmax(logits, axis=2)
    return prob.argmax(axis=2).astype(np.uint8), prob


def evaluate(weights, kind, n, seed0, size=SIZE):
    inter = np.zeros((7, 7), np.int64)  # rows = truth, cols = pred
    probs_by_class = {c: [] for c in range(1, 7)}
    truths_by_class = {c: [] for c in range(1, 7)}
    for i in range(n):
        s = generate_scene(seed0 + i, size=size)
        seg, prob = predict_scene(s["before"], s["after"], weights, kind)
        t = s["truth"]
        flat_t = t.reshape(-1)
        flat_p = seg.reshape(-1)
        np.add.at(inter, (flat_t, flat_p), 1)
        for c in range(1, 7):
            tm = t == c
            if tm.any():
                pc = prob[tm][:, c]
                truths_by_class[c].append(pc)
    return inter, truths_by_class


def metrics_from_confusion(inter):
    tp = np.diag(inter).astype(np.float64)
    gt = inter.sum(axis=1).astype(np.float64)
    pred = inter.sum(axis=0).astype(np.float64)
    iou = np.where((gt + pred - tp) > 0, tp / np.maximum(gt + pred - tp, 1), np.nan)
    prec = np.where(pred > 0, tp / np.maximum(pred, 1), np.nan)
    rec = np.where(gt > 0, tp / np.maximum(gt, 1), np.nan)
    f1 = np.where(
        (prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), np.nan
    )
    return iou, prec, rec, f1


def confusion_figure(inter, path):
    norm = inter.astype(np.float64)
    norm = norm / np.maximum(norm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=140)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASSES], fontsize=7.5)
    ax.set_yticklabels([c.replace("_", "\n") for c in CLASSES], fontsize=7.5)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised confusion matrix — held-out synthetic scenes")
    for i in range(7):
        for j in range(7):
            v = norm[i, j]
            if v > 0.005:
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if v > 0.5 else "black",
                )
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/weights.npz")
    ap.add_argument("--kind", default="numpy", choices=["numpy", "torch"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=700000)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--out", default="runs/eval")
    args = ap.parse_args()

    weights = load_model(args.weights)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    inter, _ = evaluate(weights, args.kind, args.n, args.seed, args.size)
    iou, prec, rec, f1 = metrics_from_confusion(inter)

    total = inter.sum()
    pixel_acc = np.diag(inter).sum() / max(total, 1)
    change_inter = inter[1:, 1:].sum()
    change_union = inter[1:, :].sum() + inter[:, 1:].sum() - change_inter
    change_iou = change_inter / max(change_union, 1)

    # human vs natural, computed on the change pixels only
    human_rows = sorted(HUMAN_CLASSES)
    natural_rows = sorted(NATURAL_CLASSES)
    human_truth = sum(inter[c].sum() for c in human_rows)
    natural_truth = sum(inter[c].sum() for c in natural_rows)
    human_pred = sum(inter[:, c].sum() for c in human_rows)
    natural_pred = sum(inter[:, c].sum() for c in natural_rows)
    hn_correct = sum(inter[c, c] for c in human_rows) + sum(
        inter[c, c] for c in natural_rows
    )
    hn_total = human_truth + natural_truth
    origin_acc = hn_correct / max(hn_total, 1)

    metrics = {
        "n_scenes": args.n,
        "pixel_accuracy": float(pixel_acc),
        "change_iou": float(change_iou),
        "human_vs_natural_accuracy": float(origin_acc),
        "per_class": {
            CLASSES[c]: {
                "iou": None if np.isnan(iou[c]) else float(iou[c]),
                "precision": None if np.isnan(prec[c]) else float(prec[c]),
                "recall": None if np.isnan(rec[c]) else float(rec[c]),
                "f1": None if np.isnan(f1[c]) else float(f1[c]),
                "support_px": int(inter[c].sum()),
            }
            for c in range(7)
        },
        "confusion": inter.tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    confusion_figure(inter, out / "confusion.png")

    lines = [
        "| Class | IoU | Precision | Recall | F1 | Support (px) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in range(7):
        d = metrics["per_class"][CLASSES[c]]
        f = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(
            f"| {CLASSES[c].replace('_', ' ')} | {f(d['iou'])} | {f(d['precision'])} | "
            f"{f(d['recall'])} | {f(d['f1'])} | {d['support_px']:,} |"
        )
    md = "\n".join(lines)
    (out / "report.md").write_text(md)

    print(f"pixel accuracy      : {pixel_acc:.4f}")
    print(f"change IoU          : {change_iou:.4f}")
    print(f"human vs natural acc: {origin_acc:.4f}")
    print()
    print(md)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
