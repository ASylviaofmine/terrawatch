"""Build the figures used in the README.

    python tools/make_figures.py

Writes into docs/figures/:
    pipeline.png        the four-step pipeline diagram
    examples.png        before / after / detected for each change type
    class_colour_key.png  the legend used in the overlay
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from src import infer  # noqa: E402
from src.synth import CLASSES  # noqa: E402

SAMPLES = ROOT / "docs" / "samples"
FIGS = ROOT / "docs" / "figures"

CLASS_COLORS = {
    1: "#e83e2c",
    2: "#f58a21",
    3: "#d6be3c",
    4: "#f0e25c",
    5: "#348ce2",
    6: "#9660c8",
}
CLASS_LABELS = {
    1: "Deforestation",
    2: "Urban expansion",
    3: "Excavation",
    4: "New road",
    5: "Flooding",
    6: "Wildfire scar",
}

ORDER = ["deforestation", "urban_expansion", "excavation", "new_road", "flooding", "wildfire"]


def _overlay(before, seg):
    arr = np.asarray(Image.fromarray(before).copy())
    for c in range(1, 7):
        m = seg == c
        if m.any():
            arr[m] = (0.40 * arr[m] + 0.60 * np.array(
                tuple(int(CLASS_COLORS[c][i : i + 2], 16) for i in (1, 3, 5)), np.float32
            )).astype(np.uint8)
    return arr


def pipeline_figure(path):
    fig, ax = plt.subplots(figsize=(11, 2.5), dpi=150)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 2.5)
    ax.axis("off")

    boxes = [
        (0.15, "Two acquisitions\n256 x 256 RGB", "#1f2937", "#e5e7eb"),
        (2.5, "Stack channel-wise\n6 x 256 x 256", "#1f2937", "#e5e7eb"),
        (4.85, "3 x conv3x3 + ReLU\n1 x conv1x1 head", "#0e7490", "#e5e7eb"),
        (7.2, "7-class logits\nper pixel", "#1f2937", "#e5e7eb"),
        (9.15, "Change map\n+ report", "#166534", "#e5e7eb"),
    ]
    for x, label, fc, tc in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.8), 1.85, 0.95, boxstyle="round,pad=0.06", fc=fc, ec="none"
            )
        )
        ax.text(x + 0.925, 1.275, label, ha="center", va="center", color=tc, fontsize=8.4)
    for x in (2.0, 4.35, 6.7, 8.65):
        ax.annotate(
            "",
            xy=(x + 0.46, 1.275),
            xytext=(x, 1.275),
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.6),
        )
    ax.text(
        5.5,
        2.15,
        "One forward pass — 7 classes, ~5.9k parameters",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def examples_figure(weights, path):
    names = ORDER
    fig, axes = plt.subplots(3, len(names), figsize=(2.3 * len(names), 7.1), dpi=140)
    for j, name in enumerate(names):
        before = np.asarray(Image.open(SAMPLES / f"{name}_before.png").convert("RGB"))
        after = np.asarray(Image.open(SAMPLES / f"{name}_after.png").convert("RGB"))
        seg, _ = infer.predict(before, after, weights)
        ov = _overlay(before, seg)

        for i, (im, title) in enumerate(
            ((before, "before"), (after, "after"), (ov, "detected"))
        ):
            ax = axes[i, j]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(name.replace("_", " "), fontsize=8.5, pad=6)
            if j == 0:
                ax.set_ylabel(title, fontsize=9)
            for s in ax.spines.values():
                s.set_edgecolor("#cbd5e1")
    fig.suptitle(
        "Held-out synthetic scenes: input pair and the model's detection",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/weights.npz")
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    pipeline_figure(FIGS / "pipeline.png")
    print("wrote pipeline.png")

    wpath = ROOT / args.weights
    if wpath.exists():
        weights = infer.load_weights(wpath)
        examples_figure(weights, FIGS / "examples.png")
        print("wrote examples.png")
    else:
        print("! weights missing, skipped examples.png")


if __name__ == "__main__":
    main()
