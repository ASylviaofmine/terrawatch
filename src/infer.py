"""Pure-numpy inference for the change-detection network.

The deployed web app must not depend on PyTorch (a 2 GB install for a
6 MB model is not a sane deployment), so the trained weights are exported to
a .npz and the forward pass is re-implemented here with numpy alone.

This file is deliberately dependency-free: numpy only. `tools/selftest.py`
asserts that its output matches the PyTorch model bit-for-bit (to 1e-5), so
the re-implementation is verified rather than assumed.

Architecture (identical to src/model.py):
    conv3x3(6 -> 16) -> ReLU -> conv3x3(16 -> 16) -> ReLU
    -> conv3x3(16 -> 16) -> ReLU -> conv1x1(16 -> 7)
"""

import numpy as np

N_CLASSES = 7
MEAN = 0.5
STD = 0.25


def _conv3x3(x, w, b):
    """x: (H, W, C)  w: (O, C, 3, 3)  ->  (H, W, O), zero padding."""
    h, w_, c = x.shape
    o = w.shape[0]
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    out = np.zeros((h, w_, o), np.float64)
    for dy in range(3):
        for dx in range(3):
            out += np.tensordot(
                xp[dy:dy + h, dx:dx + w_, :], w[:, :, dy, dx], axes=([2], [1])
            )
    return out + b


def _conv1x1(x, w, b):
    """x: (H, W, C)  w: (O, C)  ->  (H, W, O)."""
    out = np.tensordot(x, w, axes=([2], [1]))
    return out + b


def _relu(x):
    return np.maximum(x, 0.0)


def load_weights(path):
    z = np.load(path)
    return {k: z[k].astype(np.float64) for k in z.files}


def prepare(before_u8, after_u8):
    """Stack the two RGB frames into the 6-channel network input."""
    b = np.asarray(before_u8, np.float32) / 255.0
    a = np.asarray(after_u8, np.float32) / 255.0
    x = np.concatenate([b, a], axis=2)
    return (x - MEAN) / STD


def forward(x, p):
    """Run the network on a 6-channel (H, W, 6) input."""
    h = _relu(_conv3x3(x, p["W1"], p["b1"]))
    h = _relu(_conv3x3(h, p["W2"], p["b2"]))
    h = _relu(_conv3x3(h, p["W3"], p["b3"]))
    return _conv1x1(h, p["W4"], p["b4"])


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def predict(before_u8, after_u8, weights):
    """Return (argmax class map uint8, probability array float64)."""
    x = prepare(before_u8, after_u8)
    logits = forward(x, weights)
    prob = softmax(logits, axis=2)
    return prob.argmax(axis=2).astype(np.uint8), prob
