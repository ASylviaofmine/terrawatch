"""Generate a quick checkpoint without waiting for a full training run.

    python tools/quick_train.py --steps 600

Used as a safety net during the hackathon: it produces a usable
`runs/weights.npz` in a couple of minutes so the demo, the figures and the
video can all be produced even if the long run has not finished.

The full run (`tools/train.py`) is what the README quotes; this script is
a fallback, not the result.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import ChangeNet, export_weights  # noqa: E402
from tools.train import evaluate, make_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-seed", type=int, default=100000)
    ap.add_argument("--val-seed", type=int, default=900000)
    ap.add_argument("--val-n", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChangeNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.4
    )

    out = ROOT / "runs"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    seed = args.train_seed
    for step in range(1, args.steps + 1):
        x, y = make_batch(seed, args.batch, args.size)
        seed += args.batch
        x, y = x.to(device), y.to(device)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step}/{args.steps} loss {loss.item():.4f} "
                  f"{time.time() - t0:.0f}s", flush=True)

    torch.save(model.state_dict(), out / "model.pt")
    export_weights(model, out / "weights.npz")
    m = evaluate(model, args.val_seed, args.val_n, args.size, device)
    (out / "metrics_quick.json").write_text(json.dumps(m, indent=2))
    print(json.dumps({k: v for k, v in m.items() if k != "args"}, indent=2))


if __name__ == "__main__":
    main()
