"""Evaluate a checkpoint on the validation split.

    python val.py --weights runs/train/exp/weights/best.pt              # NMS-free one-to-one head
    python val.py --weights runs/train/exp/weights/best.pt --nms        # one-to-many head + NMS
"""

import argparse
import time

import torch
from tqdm import tqdm

from yolo26.data import YOLODataset, build_loader
from yolo26.metrics import DetMetrics
from yolo26.model import build_model
from yolo26.utils import select_device


@torch.no_grad()
def evaluate(model, loader, nc, device, branch="one2one", conf=0.001, iou=0.7, half=False):
    model.eval()
    metrics = DetMetrics(nc)
    t_inf, n_img = 0.0, 0
    for imgs, targets, _ in tqdm(loader, desc=f"val[{branch}]", leave=False):
        imgs = imgs.to(device, non_blocking=True).float() / 255
        if half:
            imgs = imgs.half()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(imgs)
        preds = model.head.predict(out, branch=branch, conf=conf, iou=iou)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_inf += time.perf_counter() - t0
        n_img += imgs.shape[0]
        metrics.update(preds, targets)
    res = metrics.compute()
    res["ms_per_img"] = 1000 * t_inf / max(n_img, 1)
    return res


def print_results(res, names):
    print(f"{'class':>12} {'labels':>8} {'P':>7} {'R':>7} {'mAP50':>7} {'mAP50-95':>9}")
    total = sum(v[0] for v in res["per_class"].values())
    print(f"{'all':>12} {total:>8} {res['P']:>7.3f} {res['R']:>7.3f} {res['mAP50']:>7.3f} {res['mAP50-95']:>9.3f}")
    for c, (n, p, r, ap50, ap) in res["per_class"].items():
        print(f"{names[c]:>12} {n:>8} {p:>7.3f} {r:>7.3f} {ap50:>7.3f} {ap:>9.3f}")
    print(f"inference + post-processing: {res['ms_per_img']:.2f} ms / image")


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default=None, help="override data.root stored in the checkpoint")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="")
    ap.add_argument("--nms", action="store_true", help="use the one-to-many head + NMS instead of end-to-end")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()

    device = select_device(args.device)
    model, cfg = load_checkpoint(args.weights, device)
    root = args.data_root or cfg["data"]["root"]
    ds = YOLODataset(root, cfg["data"][args.split] if args.split in cfg["data"] else args.split,
                     cfg["train"]["imgsz"], cfg["model"]["in_channels"], augment=False)
    loader = build_loader(ds, args.batch, 0, shuffle=False)
    branch = "one2many" if (args.nms or not cfg["model"]["end2end"]) else "one2one"
    res = evaluate(model, loader, len(cfg["data"]["names"]), device, branch, args.conf, args.iou)
    print(f"\n{args.weights}  ({branch}{' + NMS' if branch == 'one2many' else ', NMS-free'})")
    print_results(res, cfg["data"]["names"])


if __name__ == "__main__":
    main()
