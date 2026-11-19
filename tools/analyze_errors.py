"""Where does the model go wrong?  Confusion matrix + error types + the worst validation images.

    python tools/analyze_errors.py --weights runs/train/yolo26n/weights/best.pt
    python tools/analyze_errors.py --weights runs/train/yolo26n/weights/best.pt --nms --conf 0.3 --top 50

Outputs (next to the weights, in runs/train/<name>/analysis[_nms]/):
    confusion_matrix.png   rows = predicted class, columns = true class, + "background" row/column
    error_summary.txt      error types per class and recall by object height (KITTI easy/moderate/hard heights)
    errors/*.png           the val images with the most mistakes, drawn on the original colour image
    errors_top.png         the 8 worst images stacked in one picture (quick look in Colab)

Colours in the error images:
    green  thin  = ground truth                 blue   = correct detection
    red         = false positive (background)   purple = right object, wrong class
    orange thick = missed object (false negative)
    yellow      = right class but box too loose (0.1 <= IoU < 0.5)
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from val import load_checkpoint  # noqa: E402
from yolo26.data import YOLODataset, build_loader  # noqa: E402
from yolo26.utils import box_iou_matrix, select_device  # noqa: E402

C_GT, C_TP, C_FP, C_CLS, C_MISS, C_LOC = (0, 200, 0), (255, 140, 0), (40, 40, 230), (200, 60, 200), (0, 165, 255), (0, 220, 255)
HEIGHT_BINS = [(0, 25, "<25px (hard 미만)"), (25, 40, "25-40px (moderate)"), (40, 1e9, ">=40px (easy)")]


def confusion_update(cm, pred, gt, nc, iou_thr=0.5):
    """Ultralytics-style matrix: one-to-one matching at IoU >= thr regardless of class."""
    if len(gt) == 0:
        for c in pred[:, 5].long().tolist():
            cm[c, nc] += 1
        return
    if len(pred) == 0:
        for c in gt[:, 0].long().tolist():
            cm[nc, c] += 1
        return
    iou = box_iou_matrix(gt[:, 1:], pred[:, :4]).cpu().numpy()
    gi, pi = np.nonzero(iou >= iou_thr)
    used_g, used_p = set(), set()
    for k in np.argsort(-iou[gi, pi]):
        g, p = gi[k], pi[k]
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        cm[int(pred[p, 5]), int(gt[g, 0])] += 1
    for p in range(len(pred)):
        if p not in used_p:
            cm[int(pred[p, 5]), nc] += 1
    for g in range(len(gt)):
        if g not in used_g:
            cm[nc, int(gt[g, 0])] += 1


def classify_errors(pred, gt):
    """Label every prediction and every GT of one image. Predictions are processed in confidence order."""
    p_type = ["fp"] * len(pred)
    p_match = [-1] * len(pred)
    g_hit = [False] * len(gt)
    if len(pred) and len(gt):
        iou = box_iou_matrix(pred[:, :4], gt[:, 1:]).cpu().numpy()  # (P, G)
        for p in np.argsort(-pred[:, 4].cpu().numpy()):
            same = (gt[:, 0] == pred[p, 5]).cpu().numpy()
            iou_same = np.where(same, iou[p], 0)
            iou_other = np.where(~same, iou[p], 0)
            g = int(iou_same.argmax())
            if iou_same[g] >= 0.5 and not g_hit[g]:
                p_type[p], p_match[p], g_hit[g] = "tp", g, True
            elif iou_same[g] >= 0.5:
                p_type[p] = "duplicate"
            elif iou_other.max() >= 0.5:
                p_type[p], p_match[p] = "wrong_class", int(iou_other.argmax())
            elif iou_same[g] >= 0.1:
                p_type[p], p_match[p] = "localization", g
    return p_type, p_match, g_hit


def plot_confusion(cm, names, path, title):
    labels = list(names) + ["background"]
    col_sum = cm.sum(0, keepdims=True)
    norm = cm / np.maximum(col_sum, 1)
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2, 1.2 * len(labels) + 1.5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == len(labels) - 1 and j == len(labels) - 1:
                continue
            ax.text(j, i, f"{norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=10)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("True class")
    ax.set_ylabel("Predicted class")
    ax.set_title(title + "\n(each column sums to 1; number in brackets = count)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def to_orig(box, r, left, top):
    x1, y1, x2, y2 = box
    return int((x1 - left) / r), int((y1 - top) / r), int((x2 - left) / r), int((y2 - top) / r)


def draw(img_path, pred, gt, p_type, g_hit, names, geom, n_err):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    r, left, top = geom
    font = cv2.FONT_HERSHEY_SIMPLEX
    for g in range(len(gt)):
        b = to_orig(gt[g, 1:].tolist(), r, left, top)
        missed = not g_hit[g]
        cv2.rectangle(img, b[:2], b[2:], C_MISS if missed else C_GT, 3 if missed else 1)
        if missed:
            cv2.putText(img, f"MISS {names[int(gt[g, 0])]}", (b[0], b[3] + 14), font, 0.45, C_MISS, 1, cv2.LINE_AA)
    style = {"tp": (C_TP, ""), "fp": (C_FP, "FP "), "duplicate": (C_FP, "DUP "), "wrong_class": (C_CLS, "CLS? "),
             "localization": (C_LOC, "LOC ")}
    for p in range(len(pred)):
        color, tag = style[p_type[p]]
        b = to_orig(pred[p, :4].tolist(), r, left, top)
        cv2.rectangle(img, b[:2], b[2:], color, 2)
        cv2.putText(img, f"{tag}{names[int(pred[p, 5])]} {float(pred[p, 4]):.2f}", (b[0], max(b[1] - 4, 12)),
                    font, 0.45, color, 1, cv2.LINE_AA)
    banner = f"{Path(img_path).name}  errors={n_err}   green=GT blue=correct red=FP purple=wrong class orange=missed yellow=loose box"
    cv2.rectangle(img, (0, 0), (img.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(img, banner, (5, 14), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold for counting a detection")
    ap.add_argument("--nms", action="store_true", help="analyse the one-to-many head + NMS instead of end-to-end")
    ap.add_argument("--top", type=int, default=30, help="how many worst images to save")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = select_device(args.device)
    model, cfg = load_checkpoint(args.weights, device)
    names, nc = cfg["data"]["names"], len(cfg["data"]["names"])
    imgsz = tuple(cfg["train"]["imgsz"])
    branch = "one2many" if (args.nms or not cfg["model"]["end2end"]) else "one2one"
    root = args.data_root or cfg["data"]["root"]
    ds = YOLODataset(root, cfg["data"].get(args.split, args.split), imgsz, cfg["model"]["in_channels"], augment=False)
    loader = build_loader(ds, args.batch, 0, shuffle=False)
    out = Path(args.weights).resolve().parent.parent / ("analysis_nms" if branch == "one2many" else "analysis")
    (out / "errors").mkdir(parents=True, exist_ok=True)

    cm = np.zeros((nc + 1, nc + 1))
    err = defaultdict(lambda: defaultdict(int))  # err[class][type]
    h_tot = np.zeros((nc, len(HEIGHT_BINS)))
    h_hit = np.zeros((nc, len(HEIGHT_BINS)))
    per_image = []

    for imgs, targets, paths in loader:
        x = imgs.to(device).float() / 255
        preds = model.head.predict(model(x), branch=branch, conf=args.conf)
        for i, (pred, path) in enumerate(zip(preds, paths)):
            pred = pred.cpu()
            gt = targets[targets[:, 0] == i][:, 1:]
            confusion_update(cm, pred, gt, nc)
            p_type, _, g_hit = classify_errors(pred, gt)

            h0, w0 = cv2.imread(path, cv2.IMREAD_UNCHANGED).shape[:2]
            r = min(imgsz[0] / h0, imgsz[1] / w0)
            left, top = (imgsz[1] - round(w0 * r)) // 2, (imgsz[0] - round(h0 * r)) // 2

            for p, t in enumerate(p_type):
                err[int(pred[p, 5])][t] += 1
            for g, hit in enumerate(g_hit):
                c = int(gt[g, 0])
                if not hit:
                    err[c]["missed"] += 1
                h = float(gt[g, 4] - gt[g, 2]) / r  # height in original pixels
                b = next(k for k, (lo, hi, _) in enumerate(HEIGHT_BINS) if lo <= h < hi)
                h_tot[c, b] += 1
                h_hit[c, b] += hit
            n_err = sum(t != "tp" for t in p_type) + sum(not h for h in g_hit)
            per_image.append((n_err, path, pred, gt, p_type, g_hit, (r, left, top)))

    # ---------------- confusion matrix
    title = f"Confusion matrix ({branch}{' + NMS' if branch == 'one2many' else ', NMS-free'}, conf>={args.conf}, IoU>=0.5)"
    plot_confusion(cm, names, out / "confusion_matrix.png", title)

    # ---------------- text summary
    types = ["tp", "missed", "fp", "wrong_class", "localization", "duplicate"]
    lines = [title, "", "Error types per class (counted at the predicted class; 'missed' at the true class)",
             f"{'class':>12} " + " ".join(f"{t:>12}" for t in types)]
    for c in range(nc):
        lines.append(f"{names[c]:>12} " + " ".join(f"{err[c][t]:>12}" for t in types))
    lines += ["", "  tp = correct, missed = object not found, fp = box on background,",
              "  wrong_class = right place but wrong class, localization = right class but IoU 0.1-0.5,",
              "  duplicate = second box on an already-found object", "",
              "Recall by object height in the original image (KITTI difficulty heights)",
              f"{'class':>12} " + " ".join(f"{lab:>22}" for _, _, lab in HEIGHT_BINS)]
    for c in range(nc):
        cells = [f"{h_hit[c, b] / h_tot[c, b]:.2f} ({int(h_hit[c, b])}/{int(h_tot[c, b])})" if h_tot[c, b] else "-"
                 for b in range(len(HEIGHT_BINS))]
        lines.append(f"{names[c]:>12} " + " ".join(f"{s:>22}" for s in cells))
    text = "\n".join(lines)
    (out / "error_summary.txt").write_text(text, encoding="utf-8")
    print(text)

    # ---------------- worst images
    per_image.sort(key=lambda t: -t[0])
    for f in (out / "errors").glob("*.png"):
        f.unlink()
    drawn = []
    for rank, (n_err, path, pred, gt, p_type, g_hit, geom) in enumerate(per_image[: args.top]):
        if n_err == 0:
            break
        img = draw(path, pred, gt, p_type, g_hit, names, geom, n_err)
        cv2.imwrite(str(out / "errors" / f"{rank + 1:03d}_{Path(path).stem}.png"), img)
        if len(drawn) < 8:
            drawn.append(cv2.resize(img, (1242, round(img.shape[0] * 1242 / img.shape[1]))))
    if drawn:
        cv2.imwrite(str(out / "errors_top.png"), np.vstack(drawn))
    print(f"\nsaved to {out}  (confusion_matrix.png, error_summary.txt, errors_top.png, errors/)")


if __name__ == "__main__":
    main()
