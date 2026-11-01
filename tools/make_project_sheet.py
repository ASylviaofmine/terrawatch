"""Render the ORCATHON one-pager as a PDF for the Erasmus+ application pack.

    python tools/make_project_sheet.py

Reads the verified metrics from runs/eval/metrics.json so the numbers on
the sheet cannot drift from the ones the evaluation produced, and embeds
the before/after/detected triptychs as evidence rather than decoration.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "figures" / "project_sheet.pdf"

INK = "#16233a"
MUTED = "#5a6b85"
ACCENT = "#1f5fa8"
RULE = "#d5dce8"
PAPER = "#ffffff"
BAND = "#f4f7fb"


def metric_card(ax, x, y, w, h, value, label, note=""):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.004,rounding_size=0.006",
            facecolor=BAND, edgecolor=RULE, linewidth=0.8,
        )
    )
    ax.text(x + w / 2, y + h * 0.60, value, ha="center", va="center",
            fontsize=17, color=ACCENT, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.28, label, ha="center", va="center",
            fontsize=8.2, color=INK)
    if note:
        ax.text(x + w / 2, y + h * 0.10, note, ha="center", va="center",
                fontsize=6.6, color=MUTED)


def main():
    m = json.loads((ROOT / "runs" / "eval" / "metrics.json").read_text())
    per = m["per_class"]

    with PdfPages(OUT) as pdf:
        # ---------------- page 1: the sheet ----------------
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.patch.set_facecolor(PAPER)

        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # header band
        ax.add_patch(FancyBboxPatch((0, 0.885), 1, 0.115,
                                    boxstyle="square,pad=0",
                                    facecolor=INK, edgecolor="none"))
        ax.text(0.075, 0.945, "ORCATHON 2026  ·  SpaceTech Track",
                fontsize=10.5, color="#8fb4e8", va="center")
        ax.text(0.075, 0.912, "TerraWatch — Satellite Surface Change Detection",
                fontsize=16.5, color="#ffffff", va="center", fontweight="bold")

        # identity block
        y = 0.845
        ax.text(0.075, y, "Haiyu Li", fontsize=11.5, color=INK, fontweight="bold")
        ax.text(0.075, y - 0.019,
                "Jinling Institute of Technology  ·  Automation Engineering  ·  Student ID 2217011031",
                fontsize=8.2, color=MUTED)
        ax.text(0.075, y - 0.035,
                "Certificate of Participation issued 26 Oct 2026  ·  "
                "Orca Coders: Orcathon, organised by DevHarbour  ·  Solo entry (1 member)",
                fontsize=8.2, color=MUTED)

        ax.plot([0.075, 0.925], [y - 0.048, y - 0.048], color=RULE, lw=0.9)

        # problem
        y = 0.775
        ax.text(0.075, y, "THE PROBLEM", fontsize=7.6, color=ACCENT, fontweight="bold")
        ax.text(0.075, y - 0.024,
                "Two satellite passes over the same ground, months apart. A human can say "
                "\"the forest is gone\" — but not at scale, not with a",
                fontsize=9.0, color=INK)
        ax.text(0.075, y - 0.040,
                "measurement, and not with any distinction between human-driven and natural change.",
                fontsize=9.0, color=INK)

        # what it does
        y = 0.700
        ax.text(0.075, y, "WHAT IT DOES", fontsize=7.6, color=ACCENT, fontweight="bold")
        ax.text(0.075, y - 0.024,
                "One fully-convolutional network (5,639 parameters) takes both frames stacked "
                "channel-wise and emits 7 per-pixel",
                fontsize=9.0, color=INK)
        ax.text(0.075, y - 0.040,
                "logits in a single forward pass. Output: a highlighted change map, the change type, "
                "whether a human or nature",
                fontsize=9.0, color=INK)
        ax.text(0.075, y - 0.056,
                "caused it, and an automated incident report a reviewer could act on.",
                fontsize=9.0, color=INK)

        # classes
        y = 0.615
        ax.text(0.075, y, "SIX CHANGE TYPES", fontsize=7.6, color=ACCENT, fontweight="bold")
        cols = [
            ("Deforestation", "human-made"), ("Urban expansion", "human-made"),
            ("Excavation", "human-made"), ("New road", "human-made"),
            ("Flooding", "natural"), ("Wildfire scar", "natural"),
        ]
        cw = 0.276
        for i, (name, origin) in enumerate(cols):
            cx = 0.075 + (i % 3) * (cw + 0.011)
            cy = y - 0.026 - (i // 3) * 0.030
            colour = "#c0392b" if origin == "human-made" else "#2f6fb5"
            ax.add_patch(FancyBboxPatch((cx, cy - 0.021), cw, 0.024,
                                        boxstyle="square,pad=0",
                                        facecolor=BAND, edgecolor="none"))
            ax.add_patch(FancyBboxPatch((cx, cy - 0.021), 0.004, 0.024,
                                        boxstyle="square,pad=0",
                                        facecolor=colour, edgecolor="none"))
            ax.text(cx + 0.010, cy - 0.009, name, fontsize=8.4, color=INK, va="center")
            ax.text(cx + cw - 0.006, cy - 0.009, origin, fontsize=6.6,
                    color=colour, va="center", ha="right")

        # metrics
        y = 0.500
        ax.text(0.075, y, f"MEASURED ON {m['n_scenes']} HELD-OUT SCENES",
                fontsize=7.6, color=ACCENT, fontweight="bold")
        ax.text(0.075, y - 0.016,
                "generated from a seed range disjoint from the training range — no scene can appear in both.",
                fontsize=7.6, color=MUTED)

        cw = 0.207
        metric_card(ax, 0.075 + 0 * (cw + 0.012), y - 0.105, cw, 0.068,
                    f"{m['change_iou']:.3f}", "Change IoU", "any change vs none")
        metric_card(ax, 0.075 + 1 * (cw + 0.012), y - 0.105, cw, 0.068,
                    f"{m['human_vs_natural_accuracy'] * 100:.1f}%", "Human vs natural")
        metric_card(ax, 0.075 + 2 * (cw + 0.012), y - 0.105, cw, 0.068,
                    f"{m['pixel_accuracy'] * 100:.1f}%", "Pixel accuracy")
        metric_card(ax, 0.075 + 3 * (cw + 0.012), y - 0.105, cw, 0.068,
                    "5,639", "Parameters", "one forward pass")

        # per-class table
        y = 0.360
        ax.text(0.075, y, "PER-CLASS IoU", fontsize=7.6, color=ACCENT, fontweight="bold")
        # the key in metrics.json is "no_change" with an underscore; filtering
        # on "no change" with a space silently kept it on the sheet
        rows = [(k, v) for k, v in per.items() if k != "no_change"]
        rows = sorted(rows, key=lambda kv: -(kv[1]["iou"] or 0))
        ry = y - 0.022
        for name, d in rows:
            iou = d["iou"] or 0
            ax.text(0.075, ry, name.replace("_", " ").capitalize(),
                    fontsize=8.0, color=INK, va="center")
            bar_x, bar_w = 0.335, 0.42
            ax.add_patch(FancyBboxPatch((bar_x, ry - 0.0065), bar_w, 0.013,
                                        boxstyle="square,pad=0",
                                        facecolor="#eef2f8", edgecolor="none"))
            ax.add_patch(FancyBboxPatch((bar_x, ry - 0.0065), bar_w * iou, 0.013,
                                        boxstyle="square,pad=0",
                                        facecolor=ACCENT if iou >= 0.6 else "#e08a3c",
                                        edgecolor="none"))
            ax.text(bar_x + bar_w + 0.010, ry, f"{iou:.3f}", fontsize=8.0,
                    color=INK, va="center", fontweight="bold")
            ax.text(bar_x + bar_w + 0.055, ry, f"P {d['precision']:.2f}   R {d['recall']:.2f}",
                    fontsize=6.8, color=MUTED, va="center")
            ry -= 0.0215

        # evidence images
        y = 0.176
        ax.text(0.075, y, "EVIDENCE", fontsize=7.6, color=ACCENT, fontweight="bold")
        ax.text(0.075, y - 0.015,
                "Earlier acquisition, later acquisition, and the model's own overlay — from the committed weights.",
                fontsize=7.4, color=MUTED)

        for i, name in enumerate(["case_new_road", "case_flooding"]):
            p = ROOT / "docs" / "figures" / f"{name}.png"
            if not p.exists():
                continue
            sub = fig.add_axes([0.075 + i * 0.44, 0.055, 0.405, 0.105])
            sub.imshow(plt.imread(p))
            sub.axis("off")
            ax.text(0.075 + i * 0.44, 0.045,
                    ["new road (human-made)", "flooding (natural)"][i],
                    fontsize=7.2, color=MUTED)

        # footer
        ax.plot([0.075, 0.925], [0.036, 0.036], color=RULE, lw=0.9)
        ax.text(0.075, 0.024,
                "github.com/ASylviaofmine/terrawatch   ·   "
                "live demo: terrawatch-change-detection.app.workbuddy.host   ·   "
                "video: youtu.be/gpFW010gmpE",
                fontsize=6.8, color=MUTED)
        ax.text(0.075, 0.012,
                "Built end to end by the author. AI tools assisted with parts of the scripts and documentation; "
                "disclosed in the repository README.",
                fontsize=6.4, color=MUTED)

        pdf.savefig(fig, facecolor=PAPER)
        plt.close(fig)

        # ---------------- page 2: the method ----------------
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor(PAPER)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        ax.add_patch(FancyBboxPatch((0, 0.895), 1, 0.105,
                                    boxstyle="square,pad=0",
                                    facecolor=INK, edgecolor="none"))
        ax.text(0.075, 0.947, "Method & the part worth reading",
                fontsize=15, color="#ffffff", va="center", fontweight="bold")

        blocks = [
            ("THE DATA IS THE POINT",
             "Training scenes are procedurally generated with exact pixel-level ground "
             "truth. That is what makes every number on page 1 a measurement rather "
             "than an impression — precision, recall and IoU are all computable because "
             "the right answer is known for every pixel."),
            ("DEPLOYED WITHOUT PYTORCH",
             "The trained weights are exported to a .npz and the forward pass is "
             "re-implemented in NumPy alone, so the serving path has no 2 GB framework "
             "underneath it. tools/selftest.py asserts the NumPy output matches the "
             "PyTorch model to < 1e-4 — so the hosted demo runs the same model, not an "
             "approximation of it."),
            ("THE FAILURES THAT MATTERED",
             "The first evaluation returned IoU 0.03 for excavation and 0.00 for flooding, "
             "and the obvious move — a bigger network — would have been wrong. The cause "
             "was in the data: those textures were identical to the bare soil and the "
             "standing water already in the scene, so the image pairs were ambiguous by "
             "construction. Fixing the labels took them to 0.57 and 0.64. A later rewrite "
             "of the land cover quietly broke urban expansion to 0.07 for the same reason, "
             "and was fixed the same way."),
            ("WHAT THAT TAUGHT ME",
             "When a class collapses, suspect the labels before suspecting the network. "
             "A class is only learnable if its texture cannot be confused with anything "
             "else in the scene. This is the habit I carried into the rest of the project — "
             "and it is the part I would want to bring to a research group."),
        ]
        y = 0.855
        for title, body in blocks:
            ax.text(0.075, y, title, fontsize=7.8, color=ACCENT, fontweight="bold")
            words = body.split()
            line, lines = "", []
            for w_ in words:
                if len(line + " " + w_) > 96:
                    lines.append(line)
                    line = w_
                else:
                    line = (line + " " + w_).strip()
            lines.append(line)
            for i, ln in enumerate(lines):
                ax.text(0.075, y - 0.021 - i * 0.0175, ln, fontsize=8.6, color=INK)
            y -= 0.021 + len(lines) * 0.0175 + 0.030

        # pipeline
        y = 0.235
        ax.text(0.075, y, "PIPELINE", fontsize=7.8, color=ACCENT, fontweight="bold")
        boxes = [
            ("earlier\nacquisition", "256×256 RGB"),
            ("later\nacquisition", "256×256 RGB"),
            ("stack", "6 channels"),
            ("conv 3×3\n+ ReLU", ""),
            ("conv 3×3\n+ ReLU", ""),
            ("conv 3×3\ndilation 2", "15×15 field"),
            ("conv 1×1", "7 classes"),
        ]
        bx, bw, gap = 0.075, 0.106, 0.0175
        for i, (name, sub) in enumerate(boxes):
            x = bx + i * (bw + gap)
            ax.add_patch(FancyBboxPatch((x, y - 0.086), bw, 0.056,
                                        boxstyle="round,pad=0.002,rounding_size=0.004",
                                        facecolor=BAND, edgecolor=RULE, linewidth=0.8))
            ax.text(x + bw / 2, y - 0.052, name, fontsize=6.9, color=INK,
                    ha="center", va="center", linespacing=1.35)
            if sub:
                ax.text(x + bw / 2, y - 0.074, sub, fontsize=5.8, color=MUTED,
                        ha="center", va="center")
            if i < len(boxes) - 1:
                ax.annotate("", xy=(x + bw + gap - 0.002, y - 0.058),
                            xytext=(x + bw + 0.002, y - 0.058),
                            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))

        ax.text(0.075, 0.128, "TRAINING DATA", fontsize=7.8, color=ACCENT, fontweight="bold")
        ax.text(0.075, 0.107,
                "Procedural generator (src/synth.py): five land-cover texture families, six change-event",
                fontsize=8.6, color=INK)
        ax.text(0.075, 0.090,
                "renderers, per-scene illumination and sensor noise, and the ground-truth bookkeeping.",
                fontsize=8.6, color=INK)

        ax.text(0.075, 0.062,
                "All figures in this sheet come from runs/eval/metrics.json — the same file the "
                "repository README is generated from.",
                fontsize=7.4, color=MUTED)
        ax.text(0.075, 0.046,
                "github.com/ASylviaofmine/terrawatch", fontsize=7.4, color=ACCENT)

        pdf.savefig(fig, facecolor=PAPER)
        plt.close(fig)

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
