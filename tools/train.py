"""Train the change-detection network.

Run from the repository root:

    python tools/train.py --steps 3000 --batch 16

Everything is seeded, so two runs with the same arguments produce the same
weights. The split is by *scene*, and every scene is generated on the fly
from a distinct seed, so no image can appear in two splits by accident.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

from src.model import ChangeNet, export_weights  # noqa: E402
from src.synth import generate_scene  # noqa: E402


def make_batch(seed, batch, size):
    """Generate `batch` synthetic scenes and stack them for the network."""
    xs, ys = [], []
    for i in range(batch):
        s = generate_scene(seed + i, size=size)
        before = s["before"].astype(np.float32) / 255.0
        after = s["after"].astype(np.float32) / 255.0
        x = np.concatenate([before, after], axis=2)
        x = (x - 0.5) / 0.25
        xs.append(x.transpose(2, 0, 1))
        ys.append(s["truth"])
    return (
        torch.from_numpy(np.stack(xs)).float(),
        torch.from_numpy(np.stack(ys)).long(),
    )


@torch.no_grad()
def evaluate(model, seed0, n, size, device, batch=8):
    """Pixel accuracy and per-class IoU over a fixed evaluation set."""
    model.eval()
    inter = np.zeros(7, np.int64)
    union = np.zeros(7, np.int64)
    hit = np.zeros(7, np.int64)
    total = np.zeros(7, np.int64)
    correct = 0
    seen = 0
    for i in range(0, n, batch):
        x, y = make_batch(seed0 + i, min(batch, n - i), size)
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1).cpu().numpy()
        truth = y.cpu().numpy()
        correct += int((pred == truth).sum())
        seen += truth.size
        for c in range(7):
            p, t = pred == c, truth == c
            inter[c] += int((p & t).sum())
            union[c] += int((p | t).sum())
            hit[c] += int((p & t).sum())
            total[c] += int(t.sum())
    model.train()
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    prec = np.where(total > 0, hit / np.maximum(total, 1), np.nan)
    # binary "did this pixel change" view -- the question the product asks
    bin_inter = int((inter[1:].sum()))
    bin_union = int((union[1:].sum()))
    return {
        "pixel_acc": correct / max(seen, 1),
        "per_class_iou": iou.tolist(),
        "per_class_precision": prec.tolist(),
        "change_iou": bin_inter / max(bin_union, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-seed", type=int, default=100000)
    ap.add_argument("--val-seed", type=int, default=900000)
    ap.add_argument("--val-n", type=int, default=64)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--out", type=str, default="runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  steps={args.steps}  batch={args.batch}  size={args.size}")

    model = ChangeNet(width=args.width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.3
    )

    out_dir = ROOT / args.out
    out_dir.mkdir(exist_ok=True)

    model.train()
    t0 = time.time()
    seed = args.train_seed
    for step in range(1, args.steps + 1):
        x, y = make_batch(seed, args.batch, args.size)
        seed += args.batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0 or step == 1:
            print(
                f"step {step:5d}/{args.steps}  loss {loss.item():.4f}  "
                f"lr {sched.get_last_lr()[0]:.5f}  "
                f"{time.time() - t0:6.1f}s",
                flush=True,
            )

    torch.save(model.state_dict(), out_dir / "model.pt")
    export_weights(model, out_dir / "weights.npz")

    metrics = evaluate(model, args.val_seed, args.val_n, args.size, device)
    metrics["params"] = int(n_params)
    metrics["train_seconds"] = round(time.time() - t0, 1)
    metrics["args"] = vars(args)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "args"}, indent=2))
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
