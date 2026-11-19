"""COCO-style detection metrics: precision, recall, mAP@0.5 and mAP@0.5:0.95 (101-point interpolation)."""

import numpy as np
import torch

from .utils import box_iou_matrix

IOU_THRS = torch.linspace(0.5, 0.95, 10)


def match_predictions(pred, gt):
    """pred (n, 6) [x1 y1 x2 y2 conf cls], gt (m, 5) [cls x1 y1 x2 y2] -> tp (n, 10) bool."""
    tp = np.zeros((pred.shape[0], len(IOU_THRS)), dtype=bool)
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return tp
    iou = box_iou_matrix(gt[:, 1:], pred[:, :4])  # (m, n)
    correct_class = gt[:, 0:1] == pred[:, 5]
    iou = (iou * correct_class).cpu().numpy()
    for k, thr in enumerate(IOU_THRS.tolist()):
        matches = np.nonzero(iou >= thr)
        if matches[0].shape[0]:
            m = np.stack(matches, 1)  # (gt, pred)
            scores = iou[matches[0], matches[1]]
            m = m[scores.argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]  # one gt per prediction
            # re-sort by IoU so that each gt keeps its best prediction
            m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
            tp[m[:, 1], k] = True
    return tp


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return np.trapezoid(np.interp(x, mrec, mpre), x) if hasattr(np, "trapezoid") else np.trapz(np.interp(x, mrec, mpre), x)


def ap_per_class(tp, conf, pred_cls, target_cls, nc, eps=1e-16):
    """Returns dict with per-class P, R, AP50, AP(0.5:0.95) and their means over classes that have labels."""
    order = np.argsort(-conf)
    tp, conf, pred_cls = tp[order], conf[order], pred_cls[order]
    ap = np.zeros((nc, tp.shape[1]))
    p_at, r_at = np.zeros(nc), np.zeros(nc)
    n_l = np.array([(target_cls == c).sum() for c in range(nc)])
    for c in range(nc):
        i = pred_cls == c
        n_p = i.sum()
        if n_p == 0 or n_l[c] == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l[c] + eps)
        precision = tpc / (tpc + fpc)
        for j in range(tp.shape[1]):
            ap[c, j] = compute_ap(recall[:, j], precision[:, j])
        # P and R at the confidence that maximises F1 (IoU 0.5)
        f1 = 2 * precision[:, 0] * recall[:, 0] / (precision[:, 0] + recall[:, 0] + eps)
        k = f1.argmax()
        p_at[c], r_at[c] = precision[k, 0], recall[k, 0]
    valid = n_l > 0
    return {
        "P": p_at[valid].mean() if valid.any() else 0.0,
        "R": r_at[valid].mean() if valid.any() else 0.0,
        "mAP50": ap[valid, 0].mean() if valid.any() else 0.0,
        "mAP50-95": ap[valid].mean() if valid.any() else 0.0,
        "per_class": {c: (int(n_l[c]), p_at[c], r_at[c], ap[c, 0], ap[c].mean()) for c in range(nc)},
    }


class DetMetrics:
    def __init__(self, nc):
        self.nc = nc
        self.stats = []  # (tp, conf, pred_cls, target_cls)

    def update(self, preds, targets):
        """preds: list of (n, 6) per image; targets: (N, 6) [img, cls, x1, y1, x2, y2] for the batch."""
        for i, pred in enumerate(preds):
            gt = targets[targets[:, 0] == i][:, 1:].to(pred.device)
            tp = match_predictions(pred, gt)
            self.stats.append((tp, pred[:, 4].cpu().numpy(), pred[:, 5].cpu().numpy(), gt[:, 0].cpu().numpy()))

    def compute(self):
        if not self.stats:
            return ap_per_class(np.zeros((0, 10)), np.zeros(0), np.zeros(0), np.zeros(0), self.nc)
        tp, conf, pcls, tcls = (np.concatenate(x, 0) for x in zip(*self.stats))
        return ap_per_class(tp, conf, pcls, tcls, self.nc)
