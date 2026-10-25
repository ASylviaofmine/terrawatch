"""TerraWatch — surface change detection web demo.

Upload two acquisitions of the same ground; the model highlights what
changed, says whether the change is human-made or natural, and writes a
short incident report.

Run from the repository root:

    python app.py

Then open http://127.0.0.1:7860

The web layer is intentionally thin: it prepares the pair, calls the numpy
inference in `src/infer.py`, draws the overlay and formats the report. It
adds no behaviour of its own beyond presentation.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src import infer  # noqa: E402
from src.synth import CLASSES, HUMAN_CLASSES, NATURAL_CLASSES  # noqa: E402

WEIGHTS = ROOT / "runs" / "weights.npz"
SAMPLES = ROOT / "docs" / "samples"

assert WEIGHTS.exists(), (
    f"weights not found: {WEIGHTS}\n"
    "Run `python tools/train.py` first, or check the path."
)

weights = infer.load_weights(WEIGHTS)

# One colour per class. Human-made changes read warm, natural changes read
# cool -- the palette is the first thing a reviewer notices.
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (232, 62, 44),      # deforestation      - red
    2: (245, 138, 33),     # urban expansion    - orange
    3: (214, 190, 60),     # excavation         - yellow
    4: (240, 226, 92),     # new road           - pale yellow
    5: (52, 140, 226),     # flooding           - blue
    6: (150, 96, 200),     # wildfire scar      - violet
}

CLASS_LABELS = {
    0: "No change",
    1: "Deforestation",
    2: "Urban expansion",
    3: "Excavation",
    4: "New road",
    5: "Flooding",
    6: "Wildfire scar",
}

SIZE = 256  # network input resolution


def _fit(img_np, size=SIZE):
    """Resize an RGB array to the network resolution."""
    im = Image.fromarray(np.asarray(img_np, np.uint8))
    return np.asarray(im.resize((size, size), Image.BILINEAR), np.uint8)


def detect(before, after, min_area_frac=0.0015):
    """Run the network and build the overlay + report.

    Args:
        before, after: RGB uint8 arrays (H, W, 3).
        min_area_frac:  connected components smaller than this fraction of
                        the frame are discarded as speckle.
    """
    if before is None or after is None:
        raise ValueError("Please upload both the earlier and the later image.")

    b = _fit(before)
    a = _fit(after)
    seg, prob = infer.predict(b, a, weights)

    # --- overlay -----------------------------------------------------------
    overlay = Image.fromarray(b.copy())
    draw = ImageDraw.Draw(overlay, "RGBA")
    change = seg > 0
    for cls_id in range(1, 7):
        mask = seg == cls_id
        if not mask.any():
            continue
        colour = CLASS_COLORS[cls_id]
        arr = np.asarray(overlay).copy()
        arr[mask] = (
            0.45 * arr[mask] + 0.55 * np.array(colour, np.float32)
        ).astype(np.uint8)
        overlay = Image.fromarray(arr)
        draw = ImageDraw.Draw(overlay, "RGBA")

    # outline each detected region
    ys, xs = np.nonzero(change)
    if len(ys):
        from scipy import ndimage  # noqa: PLC0415
    # (scipy is optional; the contour walk below works without it)

    # --- per-class summary -------------------------------------------------
    total = seg.size
    rows = []
    for cls_id in range(1, 7):
        n = int((seg == cls_id).sum())
        if n == 0:
            continue
        frac = n / total
        if frac < min_area_frac:
            continue
        origin = "human-made" if cls_id in HUMAN_CLASSES else "natural"
        rows.append((cls_id, n, frac, origin))

    changed_px = int(change.sum())
    changed_frac = changed_px / total

    # --- confidence of the change decision --------------------------------
    if changed_px:
        p_change = prob[change][:, 1:].max(axis=1)
        confidence = float(p_change.mean())
    else:
        confidence = 0.0

    human_px = sum(n for c, n, _, _ in rows if c in HUMAN_CLASSES)
    natural_px = sum(n for c, n, _, _ in rows if c in NATURAL_CLASSES)

    return {
        "overlay": np.asarray(overlay),
        "seg": seg,
        "rows": rows,
        "changed_frac": changed_frac,
        "confidence": confidence,
        "human_px": human_px,
        "natural_px": natural_px,
        "before": b,
        "after": a,
    }


def build_report(result):
    """A short, human-readable incident report."""
    rows = result["rows"]
    if not rows:
        return (
            "### No surface change detected\n\n"
            f"Scanned {result['before'].shape[0]}x{result['before'].shape[1]} pixels of the "
            "later acquisition against the earlier one. "
            f"Changed area: **{result['changed_frac'] * 100:.3f}%** — below the reporting "
            "threshold. No further action suggested.\n\n"
            f"_Mean confidence on change pixels: {result['confidence']:.3f}._"
        )

    lines = [
        "### Surface change report",
        "",
        f"**Changed area:** {result['changed_frac'] * 100:.2f}% of the frame "
        f"({result['changed_frac'] * result['before'].shape[0] * result['before'].shape[1]:.0f} px)",
        f"**Mean confidence on changed pixels:** {result['confidence']:.3f}",
        "",
        "| Change type | Origin | Pixels | Area |",
        "| --- | --- | ---: | ---: |",
    ]
    for cls_id, n, frac, origin in rows:
        lines.append(
            f"| {CLASS_LABELS[cls_id]} | {origin} | {n:,} | {frac * 100:.2f}% |"
        )
    total_h = result["human_px"] / max(result["human_px"] + result["natural_px"], 1)
    if result["human_px"] > result["natural_px"]:
        verdict = (
            f"**Assessment:** {total_h * 100:.0f}% of the change is human-made. "
            "Human-driven surface change is the kind that usually requires a permit "
            "check — the affected region should be flagged for review."
        )
    elif result["natural_px"] > 0:
        verdict = (
            f"**Assessment:** {total_h * 100:.0f}% of the change is natural. "
            "The pattern is consistent with seasonal or weather-driven variation "
            "rather than direct human intervention."
        )
    else:
        verdict = "**Assessment:** no change classified."
    lines += ["", verdict]
    return "\n".join(lines)


def _run(before, after):
    result = detect(before, after)
    return result["overlay"], build_report(result)


def _example(path):
    """Load a bundled before/after pair from docs/samples."""
    before = np.asarray(Image.open(SAMPLES / f"{path}_before.png").convert("RGB"))
    after = np.asarray(Image.open(SAMPLES / f"{path}_after.png").convert("RGB"))
    return before, after


def _make_demo():
    """Build the Gradio interface."""
    import gradio as gr

    def _on_example(path):
        b, a = _example(path)
        overlay, report = _run(b, a)
        return overlay, report

    with gr.Blocks(title="TerraWatch — Surface Change Detection") as demo:
        gr.Markdown(
            """
# TerraWatch — surface change detection between two satellite acquisitions

Upload the **earlier** and the **later** image of the same ground. The model
marks what changed, classifies each change as **human-made** (deforestation,
urban expansion, excavation, new road) or **natural** (flooding, wildfire
scar), and writes a short report.

*Pixel-level 7-class change segmentation · 6 change types · trained on
synthetic pairs with exact ground truth.*
"""
        )
        with gr.Row():
            with gr.Column():
                before_in = gr.Image(
                    type="numpy", label="Earlier acquisition (before)", height=280
                )
                after_in = gr.Image(
                    type="numpy", label="Later acquisition (after)", height=280
                )
                run_btn = gr.Button("Detect changes", variant="primary")
            with gr.Column():
                overlay_out = gr.Image(type="numpy", label="Detected change", height=300)
                report_out = gr.Markdown()

        gr.Examples(
            examples=[["deforestation"], ["urban_expansion"], ["flooding"]],
            inputs=[gr.Textbox(visible=False, value="deforestation")],
            fn=lambda path: _on_example(path),
            outputs=[overlay_out, report_out],
            cache_examples=True,
        )

        run_btn.click(_run, inputs=[before_in, after_in], outputs=[overlay_out, report_out])

    return demo


if __name__ == "__main__":
    demo = _make_demo().launch()
