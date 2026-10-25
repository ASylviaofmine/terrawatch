"""Render the submission video.

    python tools/make_demo.py

Writes docs/demo.mp4 (and docs/demo.gif). The video is generated from the
model's own outputs, so every number and every highlighted region in it is
the model's, not an artist's impression.

The video needs no screen recording and no voice-over to be useful: the
captions carry the argument, and the recording guide for adding narration
is in docs/DEMO_SCRIPT.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import infer  # noqa: E402
from src.synth import CLASSES, HUMAN_CLASSES  # noqa: E402

SAMPLES = ROOT / "docs" / "samples"
OUT = ROOT / "docs"

W, H = 1280, 720
BG = (14, 17, 23)
FG = (232, 236, 242)
MUTED = (140, 150, 165)
ACCENT = (86, 204, 242)

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


def _font(size):
    for name in (
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        p = Path(name)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                pass
    return ImageFont.load_default()


def _overlay(before, seg):
    arr = np.asarray(Image.fromarray(before).copy())
    for c in range(1, 7):
        m = seg == c
        if m.any():
            arr[m] = (0.42 * arr[m] + 0.58 * np.array(CLASS_COLORS[c], np.float32)).astype(
                np.uint8
            )
    return Image.fromarray(arr)


def _fit_panel(img, size=320):
    return Image.fromarray(np.asarray(img, np.uint8)).resize((size, size), Image.LANCZOS)


def _text(d, xy, s, font, fill=FG, anchor=None):
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def frame_title(f1, f2, title, sub=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    _text(d, (60, 70), title, f2)
    if sub:
        _text(d, (60, 128), sub, f1, MUTED)
    return img, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    weights = infer.load_weights(ROOT / "runs" / "weights.npz")

    f_title = _font(40)
    f_head = _font(30)
    f_body = _font(21)
    f_small = _font(17)

    frames = []

    def add(img, hold=1):
        for _ in range(hold):
            frames.append(img)

    # ---- shot 1: the problem -------------------------------------------
    img, d = frame_title(f_title, f_body, "TerraWatch", "Surface change detection between two satellite acquisitions")
    _text(d, (60, 250), "Two images of the same ground, months apart.", f_body, FG)
    _text(d, (60, 290), "A human cannot diff them at scale. A model can.", f_body, FG)
    _text(d, (60, 400), "ORCA CODER  ·  ORCATHON 2026  ·  SpaceTech track", f_small, MUTED)
    add(img, hold=args.fps * 2)

    # ---- shot 2..N: one sample per change type -------------------------
    order = [
        ("deforestation", "Human-made", "Forest cleared for agriculture."),
        ("urban_expansion", "Human-made", "New buildings where open ground was."),
        ("excavation", "Human-made", "Open-pit excavation, terraced and bare."),
        ("new_road", "Human-made", "Asphalt laid over bare ground."),
        ("flooding", "Natural", "Water spreading across the flood plain."),
        ("wildfire", "Natural", "Burn scar, charred and darkened."),
    ]

    for name, origin, caption in order:
        before = np.asarray(Image.open(SAMPLES / f"{name}_before.png").convert("RGB"))
        after = np.asarray(Image.open(SAMPLES / f"{name}_after.png").convert("RGB"))
        seg, prob = infer.predict(before, after, weights)
        ov = _overlay(before, seg)

        changed = (seg > 0)
        frac = changed.mean() * 100
        counts = {c: int((seg == c).sum()) for c in range(1, 7) if (seg == c).any()}

        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        _text(d, (40, 30), f"{name.replace('_', ' ').title()}", f_head, FG)
        _text(d, (40, 68), caption, f_small, MUTED)

        pb, pa, po = _fit_panel(before), _fit_panel(after), _fit_panel(np.asarray(ov))
        img.paste(pb, (40, 120))
        img.paste(pa, (400, 120))
        img.paste(po, (760, 120))
        _text(d, (40, 452), "BEFORE", f_small, MUTED)
        _text(d, (400, 452), "AFTER", f_small, MUTED)
        _text(d, (760, 452), "DETECTED", f_small, ACCENT)

        # legend
        y = 500
        for c, n in counts.items():
            col = CLASS_COLORS[c]
            d.rectangle([760, y, 780, y + 20], fill=col)
            _text(d, (792, y), f"{CLASS_LABELS[c]}", f_body, FG)
            _text(d, (1100, y), f"{n:,} px", f_small, MUTED)
            y += 30

        _text(d, (40, 500), "Changed area", f_body, MUTED)
        _text(d, (40, 530), f"{frac:.2f}%", f_head, ACCENT)
        _text(d, (40, 580), "Origin", f_body, MUTED)
        _text(d, (40, 610), origin, f_head, (232, 62, 44) if origin == "Human-made" else (52, 140, 226))

        add(img, hold=args.fps * 2)

    # ---- final: the pipeline ------------------------------------------
    img, d = frame_title(f_title, f_body, "One forward pass, four steps", None)
    steps = [
        "1.  Stack the two acquisitions channel-wise (6 channels).",
        "2.  Three 3x3 convolutions + one 1x1 head -> 7-class logits per pixel.",
        "3.  Argmax gives the change map: no-change or one of six change types.",
        "4.  Group by origin: human-made vs natural. Write the report.",
    ]
    y = 200
    for s in steps:
        _text(d, (70, y), s, f_body, FG)
        y += 52
    _text(d, (70, y + 40), "Deployed with numpy only — no PyTorch in the serving path.", f_small, MUTED)
    _text(d, (70, y + 72), "Verified equal to the PyTorch model to 1e-4 by tools/selftest.py.", f_small, MUTED)
    add(img, hold=args.fps * 3)

    # ---- write the video ------------------------------------------------
    OUT.mkdir(exist_ok=True)
    try:
        import imageio.v2 as imageio

        mp4 = OUT / "demo.mp4"
        imageio.mimsave(str(mp4), [np.asarray(f) for f in frames], fps=args.fps, quality=8)
        print(f"wrote {mp4}  ({len(frames)} frames, {len(frames) / args.fps:.1f}s)")
        if args.gif:
            gif = OUT / "demo.gif"
            imageio.mimsave(str(gif), [np.asarray(f) for f in frames], fps=args.fps, loop=0)
            print(f"wrote {gif}")
    except Exception as exc:  # pragma: no cover - offline fallback
        print("imageio unavailable (%s); falling back to PNG frames." % exc)
        fl = OUT / "frames"
        fl.mkdir(exist_ok=True)
        for i, f in enumerate(frames):
            f.save(fl / f"f{i:04d}.png")
        print(f"wrote {len(frames)} PNG frames -> {fl}")


if __name__ == "__main__":
    main()
