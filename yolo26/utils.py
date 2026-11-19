"""Box utilities, anchors, EMA and small helpers shared across the project."""

import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- misc
def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device: str = "") -> torch.device:
    """'' -> cuda if available, then mps, else cpu. Also accepts 'cpu', '0', 'cuda:0', 'mps'."""
    device = str(device).strip().lower()
    if device in ("", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device.isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def increment_path(path: str | Path) -> Path:
    """runs/train/exp -> runs/train/exp2, exp3, ... if it already exists."""
    path = Path(path)
    if not path.exists():
        return path
    for n in range(2, 10000):
        p = Path(f"{path}{n}")
        if not p.exists():
            return p
    raise RuntimeError("too many runs")


def make_divisible(x: float, divisor: int = 8) -> int:
    return int(math.ceil(x / divisor) * divisor)


# ----------------------------------------------------------------------------- anchors / boxes
def make_anchors(feats, strides, offset: float = 0.5):
    """Anchor points (grid-cell centres, in grid units) and their strides for every feature level.

    Returns:
        anchor_points: (A, 2) x, y in grid units of their own level
        stride_tensor: (A, 1)
    """
    points, stride_list = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for f, s in zip(feats, strides):
        _, _, h, w = f.shape
        sx = torch.arange(w, device=device, dtype=dtype) + offset
        sy = torch.arange(h, device=device, dtype=dtype) + offset
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack((xx, yy), -1).view(-1, 2))
        stride_list.append(torch.full((h * w, 1), s, dtype=dtype, device=device))
    return torch.cat(points), torch.cat(stride_list)


def dist2bbox(distance, anchor_points, dim: int = -1):
    """ltrb distances -> xyxy boxes."""
    lt, rb = distance.chunk(2, dim)
    return torch.cat((anchor_points - lt, anchor_points + rb), dim)


def bbox2dist(anchor_points, bbox, reg_max: float | None = None):
    """xyxy boxes -> ltrb distances (optionally clamped for DFL bins)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    d = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    return d.clamp_(0, reg_max - 0.01) if reg_max is not None else d


def xywhn2xyxy(x, w, h):
    """Normalized cx, cy, w, h -> pixel x1, y1, x2, y2 (numpy)."""
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2)
    return y


def bbox_iou(box1, box2, CIoU: bool = False, eps: float = 1e-7):
    """Element-wise IoU / CIoU between two sets of xyxy boxes with broadcastable shapes (..., 4)."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = (b1_x2 - b1_x1).clamp(min=eps), (b1_y2 - b1_y1).clamp(min=eps)
    w2, h2 = (b2_x2 - b2_x1).clamp(min=eps), (b2_y2 - b2_y1).clamp(min=eps)

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if not CIoU:
        return iou

    # CIoU = IoU - (centre distance^2 / enclosing diagonal^2) - alpha * v   (Zheng et al., 2020)
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def box_iou_matrix(box1, box2, eps: float = 1e-7):
    """Pairwise IoU (N, M) between xyxy boxes."""
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


# ----------------------------------------------------------------------------- EMA
class ModelEMA:
    """Exponential moving average of model weights (used for validation and the saved checkpoint)."""

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: int = 2000):
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.updates = 0
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / tau))

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.updates += 1
        d = self.decay_fn(self.updates)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 - d) * msd[k].detach()
