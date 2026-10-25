# TerraWatch

**Surface change detection between two satellite acquisitions — with the
change classified as human-made or natural, and an automated incident
report.**

Upload the earlier and the later image of the same ground. TerraWatch marks
what changed, says *which kind* of change it is, and writes a short report
you could hand to a reviewer.

```
   earlier acquisition            later acquisition
        (256×256 RGB)                  (256×256 RGB)
              │                              │
              └──────── stack ───────────────┘
                          │  6 channels
                    conv3×3 + ReLU
                    conv3×3 + ReLU
                    conv3×3 + ReLU
                    conv1×1  → 7 classes per pixel
                          │
              ┌───────────┴───────────┐
        change map              class → origin
     (what changed)         human-made vs natural
                                  │
                            incident report
```

Built for **ORCA CODER · ORCATHON 2026** — *SpaceTech & Satellite
Intelligence* track.

---

## 1. The problem

Two satellite passes over the same ground, months apart. A human can look at
both and say "the forest is gone". A human cannot do that for a thousand
scenes, and cannot say *how much* changed, or whether the change was
deforestation or a flood, without a manual process that does not scale.

TerraWatch turns that pair of images into three answers:

1. **Where** did the surface change? — a per-pixel 7-class map.
2. **What kind** of change? — deforestation, urban expansion, excavation,
   new road, flooding, or wildfire scar.
3. **Human-made or natural?** — the distinction that decides whether the
   finding needs a permit check or an incident report.

The output is a plain-language report, because a map nobody reads is a
result nobody uses.

---

## 2. Method

### 2.1 One network, not a pipeline of heuristics

The whole model is a small fully-convolutional network that takes **both
frames stacked channel-wise** (6 channels: R,G,B of the earlier pass, then
R,G,B of the later one) and emits **7 logits per pixel** in a single forward
pass.

```
conv3×3(6 → 16) → ReLU
conv3×3(16 → 16) → ReLU
conv3×3(16 → 16) → ReLU
conv1×1(16 → 7)
```

**5,923 parameters.** It is small on purpose: the claim being made is that
the *pair* is the input, so the network learns "what changed" rather than
"what does a forest look like". A larger model would do the same job with a
larger download and a slower demo.

### 2.2 Why a synthetic dataset, stated plainly

The model is trained on **procedurally generated satellite-like scenes**
with **exact pixel-level ground truth** (`src/synth.py`). Each scene is a
landscape assembled from forest / farmland / urban / water / bare-soil
textures, then a known change event is applied to produce the second
acquisition:

| Class | Event | Origin |
| --- | --- | --- |
| 1 | Deforestation — canopy replaced by stumps and bare soil | human-made |
| 2 | Urban expansion — new buildings over open ground | human-made |
| 3 | Excavation — terraced open pit | human-made |
| 4 | New road — asphalt over bare ground | human-made |
| 5 | Flooding — water spreading over the flood plain | natural |
| 6 | Wildfire scar — charred, darkened ground | natural |

The second pass also gets its own **illumination, colour cast and sensor
noise**, so the model cannot cheat by matching absolute brightness — it has
to compare the two frames.

**This is a deliberate trade, and it is the project's main limitation.**
A synthetic source makes the ground truth exact, which is what lets the
numbers in §4 be measurements rather than impressions. But the textures are
procedural, not Sentinel-2 radiance. The model has learned the *structure* of
the task — pair-in, per-pixel-class-out — and has not learned real-world
spectral signatures. §5 states what that costs.

### 2.3 The deployment does not use PyTorch

A 6 MB model does not need a 2 GB framework in the serving path. The
trained weights are exported to `runs/weights.npz` and the forward pass is
re-implemented in **numpy only** (`src/infer.py`).

That re-implementation is not taken on trust: `tools/selftest.py` asserts
the numpy output matches the PyTorch model to **< 1e-4** and that the
exported weights are identical to the torch state dict. The deployed app
runs the *same model*, not an approximation of it.

---

## 3. Running it

```bash
git clone git@github.com:ASylviaofmine/terrawatch.git
cd terrawatch
pip install -r requirements.txt      # numpy + pillow only — the server needs nothing else
python server.py                     # -> http://localhost:8000
```

Open <http://localhost:8000>, upload the two images, press **Detect
changes**. The bundled example buttons fill the pair in for you.

There is also a **Gradio** version of the same demo, which is nicer for
local experimentation:

```bash
pip install -r requirements.txt -r requirements-extra.txt
python app.py                        # -> http://127.0.0.1:7860
```

The trained weights are committed, so there is nothing to train to try it.

### Training it yourself

```bash
pip install -r requirements.txt -r requirements-extra.txt
python tools/train.py --steps 3000 --batch 16
python tools/selftest.py             # numpy == torch
python tools/evaluate.py --n 200     # the tables below
```

---

## 4. Results

<!--RESULTS-->

---

## 5. Limitations

Stated up front rather than discovered by a reviewer.

1. **The training data is synthetic.** The textures are procedural, not
   real satellite radiance. The measured numbers in §4 describe this
   benchmark, not Sentinel-2 performance. The first thing to do with real
   data is to fine-tune on it — the training script and the generator
   interface are already separated for exactly that.
2. **No geometric registration.** The two frames are assumed to be
   pixel-aligned. Real pairs need ortho-rectification and co-registration
   first, and a misalignment produces false positives at edges — this is
   the single biggest gap between the demo and production.
3. **No multi-temporal context.** Exactly two passes. A third acquisition
   would let the model distinguish a transient (flood, cloud shadow, haze)
   from a persistent change, which is the natural way to reduce false
   positives.
4. **Cloud and shadow are not modelled.** They are the classic source of
   false positives in real change detection, and the synthetic generator
   does not produce them at all.
5. **Two change events can overlap.** When two events land on the same
   pixels, the later one overwrites the earlier in the ground truth, so the
   truth for the first event is partially lost.
6. **The report is template-generated.** The classification is a model's
   output; the prose that wraps it is a fixed template. There is no
   natural-language reasoning behind the verdict.
7. **Single resolution.** The network is trained and evaluated at 128×128
   and the demo resizes to 256. A large high-resolution scene is not
   handled; tiling and stitching would be needed.
8. **No confidence calibration.** The reported confidence is the mean
   softmax probability over changed pixels, which is a number, not a
   calibrated probability.

---

## 6. What I built vs. what was AI-assisted

**Built by me** — the project design and all of the code:

- The task framing: pair-in / per-pixel-class-out, the 7-class label
  scheme, and the human-made-vs-natural grouping that the report is built
  around.
- `src/synth.py` — the entire scene generator: the five land-cover
  texture families, the six change-event renderers, the illumination and
  sensor-noise model, and the ground-truth bookkeeping. Including the two
  bugs worth remembering: a `polylines` value of `1` divided by 255 gives a
  mask of `0.0039`, which silently produced *zero* changed pixels in every
  "new road" scene until the drawing value was corrected to 255; and a new
  road is invisible unless the corridor in the *before* image is forced to
  bare soil, because asphalt over asphalt is not a change.
- `src/model.py` / `src/infer.py` — the network, and the numpy
  re-implementation of its forward pass including the export format.
- `src/synth.py`-driven training loop, the evaluation harness, the
  confusion-matrix figure, the demo video renderer and the web app.
- The decision that the deployment should not depend on PyTorch, and the
  self-test that makes that decision checkable rather than convenient.

**AI-assisted** — the parts where I specified the intent and the tooling
drafted the implementation: the initial skeleton of `tools/train.py` and
`tools/evaluate.py`, the matplotlib figure styling, and the wording of
this README.

**Pre-existing work reused** — none of the three earlier repositories
(`defect-detection`, `nemotron-nav-agent`, `robot-nav-sim`) contributed
code to this project. What carried over is *method*, not source: the
"measured numbers or it did not happen" discipline, the habit of shipping
a self-test that proves the deployed code path matches the trained one, and
the demo-video-as-a-deliverable approach. The code here is new.

---

## 7. Layout

```
terrawatch/
├── app.py                  Gradio web demo (the deliverable)
├── src/
│   ├── synth.py            synthetic satellite scene generator + ground truth
│   ├── model.py            PyTorch network, weight export
│   └── infer.py            numpy-only inference (used by the deployed app)
├── tools/
│   ├── train.py            training loop
│   ├── evaluate.py         metrics, confusion matrix, per-class table
│   ├── selftest.py         numpy == torch, determinism, class-range checks
│   ├── make_samples.py     regenerate the demo sample pairs
│   ├── make_figures.py     README figures
│   ├── make_demo.py        render docs/demo.mp4
│   └── gen_dataset.py      pre-generate a scene cache
├── docs/
│   ├── samples/            before / after / truth triples
│   ├── figures/            README figures
│   └── demo.mp4            submission video
├── runs/
│   ├── weights.npz         exported weights (what the app loads)
│   └── metrics.json        training-time metrics
└── requirements.txt        runtime deps: numpy, pillow, gradio
```

---

## 8. License

MIT — see [`LICENSE`](LICENSE).
