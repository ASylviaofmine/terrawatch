"""The change-detection network (PyTorch).

This is the training-time twin of `src/infer.py`. The two must stay in
lock-step: same layer order, same channel counts, same weight layout. The
torch `Conv2d` weight of shape (out, in, kh, kw) maps directly onto the
numpy kernel used in `infer._conv3x3`, so exporting is a transpose-free
`np.savez`.

Architecture
    conv3x3(6 -> 16) -> ReLU
    conv3x3(16 -> 16) -> ReLU
    conv3x3(16 -> 16) -> ReLU
    conv1x1(16 -> 7)

The input is the two RGB frames stacked channel-wise (6 channels): the
network sees the *pair*, not two independent images, which is what lets one
forward pass say "this pixel changed".
"""

import numpy as np
import torch
import torch.nn as nn

N_CLASSES = 7
MEAN = 0.5
STD = 0.25


class ChangeNet(nn.Module):
    """A small fully-convolutional change detector.

    The third convolution is **dilated** (dilation=2, 5x5 kernel). That is
    not decoration: a change region can be 40 pixels across, and three
    plain 3x3 convolutions only see a 7x7 neighbourhood. The dilated layer
    widens the receptive field to 15x15 for one extra parameter cost, which
    is what lets the network recognise a *shape* (a terrace, a road, a
    burn scar) rather than just a local colour difference.

    Architecture (kept in lock-step with src/infer.py):
        conv3x3(6 -> 16)   -> ReLU
        conv3x3(16 -> 16)  -> ReLU
        conv3x3(d=2)(16 -> 16) -> ReLU
        conv1x1(16 -> 7)
    """

    def __init__(self, n_classes=N_CLASSES, width=16):
        super().__init__()
        self.c1 = nn.Conv2d(6, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)
        self.c3 = nn.Conv2d(width, width, 3, padding=2, dilation=2)
        self.head = nn.Conv2d(width, n_classes, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.c1(x))
        h = self.act(self.c2(h))
        h = self.act(self.c3(h))
        return self.head(h)


def prepare_np(before_u8, after_u8):
    """numpy version of the input transform (kept identical to infer.py)."""
    b = np.asarray(before_u8, np.float32) / 255.0
    a = np.asarray(after_u8, np.float32) / 255.0
    x = np.concatenate([b, a], axis=2)
    return (x - MEAN) / STD


def export_weights(model, path):
    """Write the trained weights into a .npz the numpy inference can read."""
    sd = model.state_dict()
    np.savez(
        path,
        W1=sd["c1.weight"].detach().cpu().numpy().astype(np.float64),
        b1=sd["c1.bias"].detach().cpu().numpy().astype(np.float64),
        W2=sd["c2.weight"].detach().cpu().numpy().astype(np.float64),
        b2=sd["c2.bias"].detach().cpu().numpy().astype(np.float64),
        W3=sd["c3.weight"].detach().cpu().numpy().astype(np.float64),
        b3=sd["c3.bias"].detach().cpu().numpy().astype(np.float64),
        W4=sd["head.weight"].detach().cpu().numpy().astype(np.float64),
        b4=sd["head.bias"].detach().cpu().numpy().astype(np.float64),
    )
