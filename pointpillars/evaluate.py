"""KITTI-style 3D / BEV AP (re-implemented, simplified — not the official devkit).

Follows the KITTI benchmark rules that matter most:
  * IoU threshold: Car 0.7, Pedestrian 0.5, Cyclist 0.5
  * Difficulty (by the ground-truth 2D box): Easy  >= 40 px, fully visible,  trunc <= 0.15
                                             Moderate >= 25 px, partly occluded, trunc <= 0.30
                                             Hard  >= 25 px, mostly occluded, trunc <= 0.50
  * GTs of the class that fail the difficulty, and neighbouring classes (Van for Car,
    Person_sitting for Pedestrian), are "don't care": matching them is neither TP nor FP.
  * Detections whose projected 2D box is shorter than the difficulty's minimum height are ignored.
  * AP with 40 recall points (AP|R40, the official metric since 2019).
Differences to the devkit: DontCare regions are not used, matching is greedy by score.

    python -m pointpillars.evaluate --weights runs/pointpillars/pp/weights/best.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .box_ops import corners_3d, rotated_iou
from .kitti import Calib, KittiLidar, build_loader, read_labels

IOU_THR = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
NEIGHBOUR = {"Car": "Van", "Pedestrian": "Person_sitting"}
DIFF = {"Easy": (40, 0, 0.15), "Moderate": (25, 1, 0.30), "Hard": (25, 2, 0.50)}  # min h, max occ, max trunc


def image_size(root, fid):
    f = Path(root) / "image_size" / f"{fid}.txt"
    if f.exists():
        h, w = map(int, f.read_text().split())
        return h, w
    return 375, 1242


def det_heights_2d(boxes, calib, img_hw):
    """Height in pixels of each 3D box projected to the image (clipped to the image)."""
    if len(boxes) == 0:
        return np.zeros(0)
    c = corners_3d(boxes).reshape(-1, 3)
    uv, depth = calib.lidar_to_img(c)
    uv = uv.reshape(len(boxes), 8, 2)
    v = np.clip(uv[..., 1], 0, img_hw[0])
    h = v.max(1) - v.min(1)
    h[(depth.reshape(len(boxes), 8) <= 0).all(1)] = 0
    return h


def ap_r40(scores, tps, n_gt):
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps, np.float64)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    ap = 0.0
    for r in np.linspace(1 / 40, 1, 40):
        p = precision[recall >= r]
        ap += p.max() if len(p) else 0.0
    return ap / 40


class KittiEval:
    def __init__(self, classes):
        self.classes = classes
        self.rec = {(c, d, m): {"s": [], "tp": [], "n": 0}
                    for c in classes for d in DIFF for m in ("bev", "3d")}

    def add(self, det_boxes, det_scores, det_labels, lab, calib, img_hw):
        det_h = det_heights_2d(det_boxes, calib, img_hw)
        gt_boxes = lab["boxes"]
        iou = {m: rotated_iou(det_boxes, gt_boxes, m) for m in ("bev", "3d")}
        for ci, cname in enumerate(self.classes):
            dmask = det_labels == ci
            is_cls = lab["names"] == cname
            neigh = lab["names"] == NEIGHBOUR.get(cname, "__none__")
            for dname, (minh, maxocc, maxtr) in DIFF.items():
                valid = is_cls & (lab["h2d"] >= minh) & (lab["occ"] <= maxocc) & (lab["trunc"] <= maxtr)
                ignore_gt = (is_cls & ~valid) | neigh
                for m in ("bev", "3d"):
                    r = self.rec[(cname, dname, m)]
                    r["n"] += int(valid.sum())
                    used = np.zeros(len(gt_boxes), bool)
                    for d in np.where(dmask)[0][np.argsort(-det_scores[dmask])]:
                        cand = (valid | ignore_gt) & ~used
                        ious = np.where(cand, iou[m][d], 0) if len(gt_boxes) else np.zeros(0)
                        g = int(ious.argmax()) if len(ious) else -1
                        if g >= 0 and ious[g] >= IOU_THR[cname]:
                            used[g] = True
                            if valid[g]:
                                r["s"].append(det_scores[d]); r["tp"].append(1)
                            continue  # matched a don't-care GT: ignored
                        if det_h[d] < minh:
                            continue  # too small for this difficulty: ignored
                        r["s"].append(det_scores[d]); r["tp"].append(0)

    def summary(self):
        res = {}
        for (c, d, m), r in self.rec.items():
            res[(c, d, m)] = ap_r40(r["s"], r["tp"], r["n"])
        return res


def format_table(res, classes):
    lines = [f"{'class':>11} {'metric':>6} {'Easy':>7} {'Mod.':>7} {'Hard':>7}"]
    for c in classes:
        for m in ("bev", "3d"):
            vals = [res[(c, d, m)] for d in DIFF]
            lines.append(f"{c:>11} {m.upper():>6} " + " ".join(f"{100 * v:7.2f}" for v in vals))
    mod3d = np.nanmean([res[(c, 'Moderate', '3d')] for c in classes])
    lines.append(f"mean 3D AP (Moderate): {100 * mod3d:.2f}")
    return "\n".join(lines), mod3d


@torch.no_grad()
def evaluate(model, ds, device, batch=4, workers=2, score_thr=0.1):
    model.eval()
    ev = KittiEval(ds.classes)
    loader = build_loader(ds, batch, workers, shuffle=False)
    for batch_d in tqdm(loader, desc="eval", leave=False):
        batch_d["feats"], batch_d["coords"] = batch_d["feats"].to(device), batch_d["coords"].to(device)
        preds = model.predict(model(batch_d), score_thr=score_thr)
        for (boxes, scores, labels), fid in zip(preds, batch_d["ids"]):
            calib = Calib(ds.root / "calib" / f"{fid}.txt")
            lab = read_labels(ds.root / "label_2" / f"{fid}.txt", calib, ds.classes)
            ev.add(boxes.cpu().numpy().astype(np.float64), scores.cpu().numpy(), labels.cpu().numpy(),
                   lab, calib, image_size(ds.root, fid))
    return ev.summary()


def main():
    from .model import PointPillars
    from yolo26.utils import select_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="")
    args = ap.parse_args()
    device = select_device(args.device)
    ck = torch.load(args.weights, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = PointPillars(cfg).to(device)
    model.load_state_dict(ck["model"])
    ds = KittiLidar(args.data_root or cfg["data_root"], args.split, cfg, augment=False)
    res = evaluate(model, ds, device)
    print(format_table(res, cfg["classes"])[0])


if __name__ == "__main__":
    main()
