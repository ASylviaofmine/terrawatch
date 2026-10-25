"""TerraWatch — standalone HTTP server (deployment entry point).

    python server.py            # listens on $PORT (default 8000)

Why this exists alongside `app.py`: the hosted demo should have as few
moving parts as possible. `app.py` depends on Gradio, whose version and
start-up behaviour can change underneath you; this server depends on
**numpy and Pillow only** and serves a self-contained single-page UI.

It is the same model, the same weights and the same code path as the
Gradio app — `src/infer.py` does the work in both.

    GET  /                 the UI
    POST /api/detect       JSON: {before: <b64 png>, after: <b64 png>}
                           -> {overlay: <b64 png>, report: <html>, stats: {...}}
"""

import base64
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import infer  # noqa: E402
from src.synth import CLASSES, HUMAN_CLASSES  # noqa: E402

WEIGHTS = ROOT / "runs" / "weights.npz"
if not WEIGHTS.exists():
    raise SystemExit(f"weights not found: {WEIGHTS}")

weights = infer.load_weights(WEIGHTS)

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
ORIGIN = {c: ("human-made" if c in HUMAN_CLASSES else "natural") for c in range(1, 7)}

SIZE = 256


def _fit(arr, size=SIZE):
    im = Image.fromarray(np.asarray(arr, np.uint8))
    return np.asarray(im.resize((size, size), Image.BILINEAR), np.uint8)


def analyse(before, after, min_area_frac=0.0015):
    """Run the model and build overlay + stats."""
    b = _fit(before)
    a = _fit(after)
    seg, prob = infer.predict(b, a, weights)

    overlay = b.astype(np.float32).copy()
    for c in range(1, 7):
        m = seg == c
        if m.any():
            overlay[m] = 0.42 * overlay[m] + 0.58 * np.array(CLASS_COLORS[c], np.float32)
    overlay = overlay.astype(np.uint8)

    total = seg.size
    changed = seg > 0
    rows = []
    for c in range(1, 7):
        n = int((seg == c).sum())
        if n == 0:
            continue
        frac = n / total
        if frac < min_area_frac:
            continue
        rows.append({"class": c, "label": CLASS_LABELS[c], "origin": ORIGIN[c],
                     "pixels": n, "fraction": frac})

    confidence = float(prob[changed][:, 1:].max(axis=1).mean()) if changed.any() else 0.0
    human = sum(r["pixels"] for r in rows if r["origin"] == "human-made")
    natural = sum(r["pixels"] for r in rows if r["origin"] == "natural")

    return {
        "overlay": overlay,
        "seg": seg,
        "rows": rows,
        "changed_fraction": float(changed.mean()),
        "confidence": confidence,
        "human_px": human,
        "natural_px": natural,
    }


def build_report(r):
    rows = r["rows"]
    if not rows:
        return (
            "<h3>No surface change detected</h3>"
            f"<p>Changed area: <b>{r['changed_fraction'] * 100:.3f}%</b> — below the "
            "reporting threshold. No further action suggested.</p>"
            f"<p class='muted'>Mean confidence on change pixels: {r['confidence']:.3f}</p>"
        )
    items = "".join(
        f"<tr><td><span class='swatch' style='background:rgb{CLASS_COLORS[x['class']]}'></span>"
        f"{x['label']}</td><td>{x['origin']}</td><td class='num'>{x['pixels']:,}</td>"
        f"<td class='num'>{x['fraction'] * 100:.2f}%</td></tr>"
        for x in rows
    )
    tot = r["human_px"] + r["natural_px"]
    hshare = (r["human_px"] / tot * 100) if tot else 0
    if r["human_px"] > r["natural_px"]:
        verdict = (
            f"<b>Assessment:</b> {hshare:.0f}% of the change is human-made. Human-driven "
            "surface change usually needs a permit check — the affected region should be "
            "flagged for review."
        )
    else:
        verdict = (
            f"<b>Assessment:</b> {hshare:.0f}% of the change is natural. The pattern is "
            "consistent with seasonal or weather-driven variation rather than direct "
            "human intervention."
        )
    return (
        "<h3>Surface change report</h3>"
        f"<p><b>Changed area:</b> {r['changed_fraction'] * 100:.2f}% of the frame &nbsp;·&nbsp; "
        f"<b>Mean confidence:</b> {r['confidence']:.3f}</p>"
        "<table><thead><tr><th>Change type</th><th>Origin</th><th class='num'>Pixels</th>"
        f"<th class='num'>Area</th></tr></thead><tbody>{items}</tbody></table>"
        f"<p>{verdict}</p>"
    )


def _b64_png(arr):
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, np.uint8)).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _unb64_png(s):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB"))


PAGE = (ROOT / "web" / "index.html").read_text(encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, json.dumps({"status": "ok", "classes": len(CLASSES)}))
        elif self.path.startswith("/samples/"):
            name = self.path.split("/")[-1]
            p = ROOT / "docs" / "samples" / name
            if not p.exists():
                self._send(404, json.dumps({"error": "not found"}))
                return
            self._send(200, p.read_bytes(), "image/png")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/detect":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
            before = _unb64_png(payload["before"])
            after = _unb64_png(payload["after"])
            r = analyse(before, after)
            out = {
                "overlay": _b64_png(r["overlay"]),
                "report": build_report(r),
                "changed_fraction": r["changed_fraction"],
                "confidence": r["confidence"],
                "rows": r["rows"],
            }
            self._send(200, json.dumps(out))
        except Exception as exc:  # keep the server alive on a bad request
            self._send(400, json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


def main():
    port = int(os.environ.get("PORT", "8000"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"TerraWatch listening on 0.0.0.0:{port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
