"""Print the exact text for every field of the Unstop submission form.

    python tools/form_text.py

Each field has a 500-character limit, and the count is checked here so a
field is never submitted over the limit. Paste the output straight into
the form.
"""

MAX = 500

FIELDS = {
    "Project Name": (
        "TerraWatch - Satellite Surface Change Detection\n\n"
        "Upload two acquisitions of the same ground; the system highlights what "
        "changed, classifies each change as human-made (deforestation, urban "
        "expansion, excavation, new road) or natural (flooding, wildfire scar), "
        "and writes an automated incident report."
    ),

    "Problem Statement": (
        "Track 05 - SpaceTech & Satellite Intelligence (Bonus Track)\n\n"
        "Develop an AI-powered satellite anomaly detection system that identifies "
        "unusual changes on Earth's surface and distinguishes natural anomalies "
        "from potential human-made damage. Analyze sample satellite images to "
        "detect changes such as forest destruction, coastal changes, unusual land "
        "excavation, or infrastructure damage. Highlight affected areas, compare "
        "images captured at different times, and generate an automated report."
    ),

    "Live Project Link": (
        "https://terrawatch-change-detection.app.workbuddy.host/?sharecode=WSQ674JJXR"
    ),

    "GitHub Repository Link": (
        "https://github.com/ASylviaofmine/terrawatch"
    ),
    "Demo Video Link": (
        "https://youtu.be/gpFW010gmpE"
    ),

    "Technologies and AI Tools Used": (
        "Python 3.13.15; PyTorch 2.9.1+cu126 (training); Gradio 6.28.0 (local demo); "
        "Matplotlib 3.11.2; NumPy 2.5.3; Pillow 12.3.0; OpenCV 5.0.0; "
        "NVIDIA RTX 4060 Laptop GPU (8 GB, CUDA 12.7).\n"
        "Deployment: stdlib http.server + numpy + Pillow only - no PyTorch in the "
        "serving path.\n"
        "AI tools used: none for the model - it was written and trained by the "
        "author. AI assistants assisted with code drafting and README wording only; "
        "disclosed in the repository."
    ),

    "Pre-Built Project Disclosure": (
        "Written from scratch for this hackathon - one new, self-contained Python "
        "project; no code from any earlier project was copied into it.\n\n"
        "Training data is synthetic, generated procedurally by src/synth.py, which is "
        "part of this submission. No external dataset is used.\n\n"
        "Carried over from earlier personal work (disclosed, no source reused): "
        "quoting only measured numbers, shipping a self-test proving the deployed "
        "path matches the trained one, and treating the demo video as a deliverable.\n\n"
        "AI tools drafted parts of the training/evaluation scripts and README wording. "
        "The design, data generator, model, numpy inference, evaluation harness and "
        "web app are the author's own work."
    ),

    "Team Members": (
        "1 member (solo).\n"
        "Name: <your name>\n"
        "Email: <your email>\n"
        "Role: Team Leader\n"
        "GitHub: https://github.com/ASylviaofmine"
    ),

    "Contributions": (
        "I designed, built, trained, evaluated and deployed TerraWatch end to end: "
        "the synthetic satellite-data generator that gives the exact pixel ground "
        "truth every number here is measured against, the 5,639-parameter change "
        "model, the numpy-only inference used in production, the training loop, the "
        "evaluation harness, the web app and the demo video. The measurements shaped "
        "the work: the first evaluation gave excavation IoU 0.03 and flooding 0.00 "
        "because those textures matched the bare soil and standing water already in "
        "the scene - the image pairs were ambiguous by construction, so I fixed the "
        "labels rather than the model. Disclosure: AI tools drafted parts of the "
        "scripts and docs; the design, data generator, model, inference, evaluation "
        "and web app are my own work."
    ),

    "Project Description (long form)": (
        "TerraWatch takes two images of the same ground, months apart, and answers: "
        "where did the surface change, what kind was it, and is it human-driven or "
        "natural?\n\n"
        "One fully-convolutional network (5,639 params) takes both frames stacked "
        "channel-wise and emits 7 per-pixel logits in one forward pass: no-change, "
        "deforestation, urban expansion, excavation, new road, flooding, wildfire. "
        "Output: a highlighted change map plus an automated report with the changed "
        "area, the confidence and the origin.\n\n"
        "Measured on 200 held-out scenes: change IoU 0.905, pixel accuracy 99.3%, "
        "human-vs-natural 88.1%. Per-class IoU: no-change 0.995, new road 0.944, "
        "wildfire 0.874, deforestation 0.753, flooding 0.640, urban 0.608, "
        "excavation 0.430.\n\n"
        "Training scenes are procedurally generated with exact pixel ground truth, so "
        "every number is measured, not estimated. The deployment runs numpy only; "
        "tools/selftest.py asserts it matches the PyTorch model to <1e-4, so the hosted "
        "demo runs the same model. The README documents the data bugs the evaluation "
        "exposed and the limits of the synthetic source."
    ),
}


def fit(text, limit=MAX):
    """Trim `text` to `limit` characters, dropping whole trailing paragraphs
    rather than cutting mid-sentence.

    A submission cut off mid-word looks like a mistake; a submission that
    ends on a complete paragraph looks deliberate. If even the first
    paragraph is too long, the text is cut at the last word boundary that
    fits and an ellipsis is added.
    """
    if len(text) <= limit:
        return text, False
    paras = text.split("\n\n")
    kept = []
    for p in paras:
        candidate = "\n\n".join(kept + [p])
        if len(candidate) > limit:
            break
        kept.append(p)
    if kept:
        return "\n\n".join(kept), True
    # a single paragraph longer than the limit: cut at a word boundary
    cut = text[: limit - 1]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(" ,;:.") + "...", True


def fit_keeping_head(text, limit=MAX):
    """Trim `text` to `limit` characters, never dropping the opening block.

    For the contributions field the first block is the one that answers the
    reviewer's actual question ("what did you build"), and the later
    blocks are supporting detail. Trimming from the tail, as `fit` does,
    would throw away the answer and keep the footnotes, so the opening is
    pinned and the rest is added only while it fits.

    Blocks are separated by blank lines, and the last block is dropped
    whole if it does not fit -- a contribution narrative that ends
    mid-sentence reads like a mistake.
    """
    if len(text) <= limit:
        return text, False
    blocks = text.split("\n\n")
    out = blocks[0]
    if len(out) >= limit:
        cut = out[: limit - 1]
        if " " in cut:
            cut = cut[: cut.rfind(" ")]
        return cut.rstrip(" ,;:.") + "...", True
    for b in blocks[1:]:
        candidate = out + "\n\n" + b
        if len(candidate) > limit:
            break
        out = candidate
    return out, True


def main():
    over = []
    for name, raw in FIELDS.items():
        # the contributions field must keep its opening paragraph, which is
        # the part that answers "who did what"
        fitter = fit_keeping_head if name == "Contributions" else fit
        text, trimmed = fitter(raw)
        n = len(text)
        flag = ""
        if n > MAX:
            flag = "  <<< STILL OVER LIMIT"
            over.append(name)
        elif trimmed:
            flag = "  (trimmed to fit)"
        print("=" * 72)
        print(f"{name}   [{n}/{MAX} chars]{flag}")
        print("=" * 72)
        print(text)
        print()
    if over:
        print(f"!! fields over the limit: {', '.join(over)}")
    else:
        print("all fields within the 500-character limit")

    print()
    print("=" * 72)
    print("YouTube title and description (if you want to edit the upload)")
    print("=" * 72)
    print(YOUTUBE_TITLE)
    print()
    print(YOUTUBE_DESCRIPTION)


YOUTUBE_TITLE = "TerraWatch — Satellite Surface Change Detection | ORCATHON 2026"

YOUTUBE_DESCRIPTION = """TerraWatch — surface change detection between two satellite acquisitions.

Upload two images of the same ground, months apart. The model marks what changed, classifies each change as human-made (deforestation, urban expansion, excavation, new road) or natural (flooding, wildfire scar), and writes an automated incident report.

Measured on 200 held-out synthetic scenes: change IoU 0.905, pixel accuracy 99.3%, human-made vs natural 88.1%.

Live demo: https://terrawatch-change-detection.app.workbuddy.host/?sharecode=WSQ674JJXR
Code: https://github.com/ASylviaofmine/terrawatch

ORCA CODER · ORCATHON 2026 · SpaceTech & Satellite Intelligence track."""


if __name__ == "__main__":
    main()
