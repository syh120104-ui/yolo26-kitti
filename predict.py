"""Run a trained model on images and save the drawn detections.

    python predict.py --weights runs/train/exp/weights/best.pt --source datasets/kitti_yolo/images/val --max 20
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from val import load_checkpoint
from yolo26.utils import increment_path, select_device

COLORS = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72)]


def letterbox(img, imgsz):
    H, W = imgsz
    h0, w0 = img.shape[:2]
    r = min(H / h0, W / w0)
    im = cv2.resize(img, (round(w0 * r), round(h0 * r)), interpolation=cv2.INTER_AREA)
    if im.ndim == 2:
        im = im[..., None]
    h, w = im.shape[:2]
    top, left = (H - h) // 2, (W - w) // 2
    out = np.full((H, W, im.shape[2]), 114, np.uint8)
    out[top : top + h, left : left + w] = im
    return out, r, left, top


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--nms", action="store_true", help="one-to-many head + NMS instead of end-to-end")
    ap.add_argument("--max", type=int, default=0, help="process at most N images (0 = all)")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = select_device(args.device)
    model, cfg = load_checkpoint(args.weights, device)
    names, imgsz, in_ch = cfg["data"]["names"], tuple(cfg["train"]["imgsz"]), cfg["model"]["in_channels"]
    branch = "one2many" if (args.nms or not cfg["model"]["end2end"]) else "one2one"

    src = Path(args.source)
    files = sorted(p for p in (src.iterdir() if src.is_dir() else [src])
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if args.max:
        files = files[: args.max]
    out_dir = increment_path(Path("runs/predict/exp"))
    out_dir.mkdir(parents=True)

    for p in files:
        orig = cv2.imread(str(p), cv2.IMREAD_COLOR)
        net_in = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) if in_ch == 1 else orig
        im, r, left, top = letterbox(net_in, imgsz)
        x = torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0).to(device).float() / 255
        det = model.head.predict(model(x), branch=branch, conf=args.conf)[0].cpu().numpy()
        for x1, y1, x2, y2, score, c in det:
            # undo letterbox -> original pixels
            x1, x2 = (x1 - left) / r, (x2 - left) / r
            y1, y2 = (y1 - top) / r, (y2 - top) / r
            color = COLORS[int(c) % len(COLORS)]
            cv2.rectangle(orig, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(orig, f"{names[int(c)]} {score:.2f}", (int(x1), max(int(y1) - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / p.name), orig)
    print(f"{len(files)} images -> {out_dir}  ({branch})")


if __name__ == "__main__":
    main()
