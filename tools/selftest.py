"""Self-test: the numpy inference must match the PyTorch model exactly.

    python tools/selftest.py

Checks
    1. numpy forward pass == torch forward pass (max abs diff < 1e-4)
    2. the exported .npz is byte-identical to the torch state dict
    3. the same input always produces the same output (determinism)
    4. every class in the class map is one the model is supposed to emit

The deployment runs the numpy implementation, so claim (1) is the thing
that makes the deployed app trustworthy: it is the *same model*, not a
re-implementation of it.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import infer  # noqa: E402
from src.model import ChangeNet  # noqa: E402
from src.synth import generate_scene  # noqa: E402

WEIGHTS = ROOT / "runs" / "weights.npz"
MODEL = ROOT / "runs" / "model.pt"


def main():
    failures = []

    if not WEIGHTS.exists() or not MODEL.exists():
        print("! runs/weights.npz or runs/model.pt missing -- run tools/train.py first")
        return 1

    import torch

    model = ChangeNet()
    model.load_state_dict(torch.load(MODEL, map_location="cpu"))
    model.eval()

    p = infer.load_weights(WEIGHTS)
    sd = model.state_dict()

    # (2) exported weights must equal the torch state dict
    pairs = [("W1", "c1.weight"), ("b1", "c1.bias"), ("W2", "c2.weight"),
             ("b2", "c2.bias"), ("W3", "c3.weight"), ("b3", "c3.bias"),
             ("W4", "head.weight"), ("b4", "head.bias")]
    for npk, tk in pairs:
        d = float(np.abs(p[npk] - sd[tk].detach().numpy()).max())
        if d > 0:
            failures.append(f"weights mismatch {npk} vs {tk}: max diff {d}")
    print(f"weights identical to torch state dict: {not failures}")

    # (1) numpy forward == torch forward
    max_diff = 0.0
    for seed in (111, 222, 333):
        s = generate_scene(seed, size=128)
        x = infer.prepare(s["before"], s["after"])
        out_np = infer.forward(x, p)

        xt = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32))
        with torch.no_grad():
            out_t = model(xt)[0].numpy().transpose(1, 2, 0)

        if out_np.shape != out_t.shape:
            failures.append(
                f"shape mismatch on seed {seed}: numpy {out_np.shape} vs torch {out_t.shape}"
            )
        d = float(np.abs(out_np - out_t).max())
        max_diff = max(max_diff, d)
        if d > 1e-4:
            failures.append(f"forward mismatch on seed {seed}: max diff {d}")
    print(f"numpy vs torch forward pass, max abs diff: {max_diff:.2e} (tol 1e-4)")

    # (3) determinism
    s = generate_scene(444, size=128)
    a1, _ = infer.predict(s["before"], s["after"], p)
    a2, _ = infer.predict(s["before"], s["after"], p)
    if not np.array_equal(a1, a2):
        failures.append("prediction is not deterministic")
    print(f"deterministic across repeated calls: {np.array_equal(a1, a2)}")

    # (4) class map stays in range
    if a1.min() < 0 or a1.max() > 6:
        failures.append(f"class map out of range: [{a1.min()}, {a1.max()}]")
    print(f"class map range: [{a1.min()}, {a1.max()}] (expected [0, 6])")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
