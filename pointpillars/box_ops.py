"""3D box utilities in the LiDAR frame.

Box format everywhere: (x, y, z, dx, dy, dz, heading)
    x forward, y left, z up (metres); (x, y, z) = box centre; dx = length, dy = width, dz = height;
    heading = yaw around z (rad), 0 = facing +x.
"""

import math

import numpy as np
import torch


def limit_period(val, offset=0.5, period=math.pi):
    return val - torch.floor(val / period + offset) * period if torch.is_tensor(val) else \
        val - np.floor(val / period + offset) * period


# ----------------------------------------------------------------------------- anchor box coding (SECOND / PointPillars)
def encode_boxes(gt, anchors):
    """Residuals the network learns: Δx/d, Δy/d, Δz/h, log size ratios, Δθ (sin trick applied in the loss)."""
    xa, ya, za, dxa, dya, dza, ra = anchors.unbind(-1)
    xg, yg, zg, dxg, dyg, dzg, rg = gt.unbind(-1)
    diag = torch.sqrt(dxa**2 + dya**2)
    return torch.stack(((xg - xa) / diag, (yg - ya) / diag, (zg - za) / dza,
                        torch.log(dxg / dxa), torch.log(dyg / dya), torch.log(dzg / dza), rg - ra), -1)


def decode_boxes(deltas, anchors):
    xa, ya, za, dxa, dya, dza, ra = anchors.unbind(-1)
    xt, yt, zt, dxt, dyt, dzt, rt = deltas.unbind(-1)
    diag = torch.sqrt(dxa**2 + dya**2)
    return torch.stack((xt * diag + xa, yt * diag + ya, zt * dza + za,
                        torch.exp(dxt) * dxa, torch.exp(dyt) * dya, torch.exp(dzt) * dza, rt + ra), -1)


# ----------------------------------------------------------------------------- IoU used for anchor matching
def nearest_bev_iou(a, b):
    """Axis-aligned BEV IoU after snapping each heading to the nearest 0/90 deg (fast, used for target assignment)."""
    def to_aabb(bx):
        r = limit_period(bx[:, 6], 0.5, math.pi).abs()
        swap = r > math.pi / 4
        dx = torch.where(swap, bx[:, 4], bx[:, 3])
        dy = torch.where(swap, bx[:, 3], bx[:, 4])
        return torch.stack((bx[:, 0] - dx / 2, bx[:, 1] - dy / 2, bx[:, 0] + dx / 2, bx[:, 1] + dy / 2), 1)

    A, B = to_aabb(a), to_aabb(b)
    lt = torch.max(A[:, None, :2], B[None, :, :2])
    rb = torch.min(A[:, None, 2:], B[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (A[:, 2:] - A[:, :2]).prod(-1)
    area_b = (B[:, 2:] - B[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-6)


def bev_aabb(boxes):
    """Enclosing axis-aligned BEV rectangle of rotated boxes (for the fast NMS used in the paper)."""
    c, s = torch.cos(boxes[:, 6]).abs(), torch.sin(boxes[:, 6]).abs()
    hw = (boxes[:, 3] * c + boxes[:, 4] * s) / 2
    hh = (boxes[:, 3] * s + boxes[:, 4] * c) / 2
    return torch.stack((boxes[:, 0] - hw, boxes[:, 1] - hh, boxes[:, 0] + hw, boxes[:, 1] + hh), 1)


# ----------------------------------------------------------------------------- exact rotated IoU (evaluation)
def bev_corners(boxes):
    """(N, 7) numpy -> (N, 4, 2) BEV corners, counter-clockwise."""
    x, y, dx, dy, r = boxes[:, 0], boxes[:, 1], boxes[:, 3], boxes[:, 4], boxes[:, 6]
    local = np.array([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]])
    pts = local[None] * np.stack((dx, dy), 1)[:, None]
    c, s = np.cos(r), np.sin(r)
    rot = np.stack((np.stack((c, -s), 1), np.stack((s, c), 1)), 1)  # (N, 2, 2)
    return np.einsum("nij,nkj->nki", rot, pts) + np.stack((x, y), 1)[:, None]


def _poly_area(p):
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))


def _clip(subject, clip):
    """Sutherland-Hodgman polygon clipping of convex polygons (both counter-clockwise)."""
    out = subject
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        inp, out = out, []
        if len(inp) == 0:
            break
        side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])  # noqa: E731
        for j in range(len(inp)):
            cur, prev = inp[j], inp[j - 1]
            sc, sp = side(cur), side(prev)
            if sc >= 0:
                if sp < 0:
                    t = sp / (sp - sc)
                    out.append(prev + t * (cur - prev))
                out.append(cur)
            elif sp >= 0:
                t = sp / (sp - sc)
                out.append(prev + t * (cur - prev))
        out = np.array(out) if len(out) else []
    return np.array(out) if len(out) else np.zeros((0, 2))


def rotated_iou(a, b, mode="bev"):
    """Exact IoU matrix (Na, Nb) between rotated boxes; mode 'bev' or '3d'."""
    na, nb = len(a), len(b)
    iou = np.zeros((na, nb), np.float32)
    if na == 0 or nb == 0:
        return iou
    ca, cb = bev_corners(a), bev_corners(b)
    # quick reject with enclosing circles
    ra = np.hypot(a[:, 3], a[:, 4]) / 2
    rb = np.hypot(b[:, 3], b[:, 4]) / 2
    dist = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    cand = np.argwhere(dist < ra[:, None] + rb[None])
    for i, j in cand:
        poly = _clip(ca[i], cb[j])
        inter = _poly_area(poly) if len(poly) >= 3 else 0.0
        if inter <= 0:
            continue
        area_a, area_b = a[i, 3] * a[i, 4], b[j, 3] * b[j, 4]
        if mode == "bev":
            iou[i, j] = inter / (area_a + area_b - inter)
        else:
            za0, za1 = a[i, 2] - a[i, 5] / 2, a[i, 2] + a[i, 5] / 2
            zb0, zb1 = b[j, 2] - b[j, 5] / 2, b[j, 2] + b[j, 5] / 2
            h = max(0.0, min(za1, zb1) - max(za0, zb0))
            inter3 = inter * h
            iou[i, j] = inter3 / (area_a * a[i, 5] + area_b * b[j, 5] - inter3)
    return iou


def corners_3d(boxes):
    """(N, 7) numpy -> (N, 8, 3) corners (for drawing / projecting to the image)."""
    bev = bev_corners(boxes)
    z0 = (boxes[:, 2] - boxes[:, 5] / 2)[:, None, None]
    z1 = (boxes[:, 2] + boxes[:, 5] / 2)[:, None, None]
    bot = np.concatenate((bev, np.broadcast_to(z0, (len(boxes), 4, 1))), 2)
    top = np.concatenate((bev, np.broadcast_to(z1, (len(boxes), 4, 1))), 2)
    return np.concatenate((bot, top), 1)
