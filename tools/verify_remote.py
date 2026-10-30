"""Verify that what is on GitHub matches what is in the local working tree.

    python tools/verify_remote.py

Two things are checked, because they fail differently:

- binary files (weights, video, web page) are compared by git blob SHA,
  which is exact;
- text files are compared after normalising line endings, because git
  stores LF while a Windows checkout writes CRLF. That difference is
  expected and is not a content change.

The point is to answer one question: if a reviewer clones the repository
on a machine that is not this one, do they get the same files that were
measured here?
"""

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "ASylviaofmine/terrawatch"
API = f"https://api.github.com/repos/{REPO}"

BINARY = [
    "runs/weights.npz",
    "runs/model.pt",
    "docs/demo.mp4",
    "docs/demo.gif",
    "web/index.html",
    "src/synth.py",
    "src/infer.py",
    "src/model.py",
    "server.py",
    "app.py",
    "requirements.txt",
    "LICENSE",
]
TEXT = ["README.md", "runs/metrics.json", "docs/SUBMISSION.md"]


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch_tree() -> dict:
    req = urllib.request.Request(
        f"{API}/git/trees/master?recursive=1", headers={"User-Agent": "verify"}
    )
    with urllib.request.urlopen(req, timeout=45) as r:
        data = json.loads(r.read())
    return {t["path"]: t["sha"] for t in data["tree"] if t["type"] == "blob"}


def main():
    remote = fetch_tree()
    ok = True

    print("binary / exact-byte files")
    for f in BINARY:
        p = ROOT / f
        if not p.exists():
            print(f"  MISSING LOCAL  {f}")
            ok = False
            continue
        mine = blob_sha(p.read_bytes())
        theirs = remote.get(f)
        same = mine == theirs
        ok = ok and same
        print(f"  {'MATCH' if same else 'DIFF '}  {f:24s} {theirs[:12] if theirs else '-'}")

    print("\ntext files (compared after LF normalisation)")
    for f in TEXT:
        p = ROOT / f
        if not p.exists():
            print(f"  MISSING LOCAL  {f}")
            ok = False
            continue
        raw = p.read_bytes()
        lf = raw.replace(b"\r\n", b"\n")
        theirs = remote.get(f)
        same = blob_sha(lf) == theirs
        ok = ok and same
        note = "" if same else "  (content differs)"
        print(f"  {'MATCH' if same else 'DIFF '}  {f:24s} {theirs[:12] if theirs else '-'}{note}")

    print()
    print("all files match" if ok else "SOME FILES DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
