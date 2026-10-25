"""Reproduce every number in the README, in one command.

    python tools/reproduce.py

Runs, in order:
    1. the self-test   -- numpy inference == PyTorch model
    2. the evaluation  -- the per-class table and the confusion matrix
    3. the figures     -- the README images
    4. the video       -- docs/demo.mp4
    5. the README fill -- the results section, from the metrics file

It writes into the working tree, so run it on a clean checkout and diff
what comes out against the committed results. If the numbers match, the
committed results are reproducible; if they do not, this says so loudly
rather than leaving it to be discovered.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STEPS = [
    ("self-test: numpy == PyTorch", [sys.executable, "tools/selftest.py"]),
    ("evaluation: 200 held-out scenes", [sys.executable, "tools/evaluate.py", "--n", "200", "--seed", "700000"]),
    ("figures", [sys.executable, "tools/make_figures.py"]),
    ("demo video", [sys.executable, "tools/make_demo.py"]),
    ("README results section", [sys.executable, "tools/fill_readme.py"]),
]


def main():
    print("=" * 72)
    print("TerraWatch — full reproduction")
    print("=" *72)
    for i, (label, cmd) in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {label}")
        print("-" * 72)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        dt = time.time() - t0
        if r.returncode != 0:
            print(f"\nFAILED at step {i} ({label}) after {dt:.1f}s")
            return r.returncode
        print(f"ok ({dt:.1f}s)")

    print("\n" + "=" * 72)
    print("all steps completed")
    print("compare the regenerated files against the committed ones with:")
    print("    git status --short")
    print("    git diff --stat")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
