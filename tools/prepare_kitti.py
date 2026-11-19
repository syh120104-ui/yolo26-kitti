"""Convert (a subset of) KITTI 2D object detection data to YOLO format.

KITTI label line:  type truncated occluded alpha x1 y1 x2 y2 h w l X Y Z ry   (pixels, original image)
YOLO label line:   cls cx cy w h                                               (normalised 0-1)

    python tools/prepare_kitti.py --kitti-root /path/to/kitti --out datasets/kitti_yolo --num-images 3000

--kitti-root can be any folder that contains (somewhere below it) an `image_2` and a `label_2` folder,
e.g. data_object_image_2/training/image_2 + data_object_label_2/training/label_2.
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2

DEFAULT_MAP = {"Car": "Car", "Pedestrian": "Pedestrian", "Cyclist": "Cyclist"}
# Optional merges, e.g. --merge Van:Car Person_sitting:Pedestrian  ("DontCare" and "Misc" are always dropped)


def find_dir(root: Path, name: str) -> Path:
    if (root / name).is_dir():
        return root / name
    hits = sorted(p for p in root.rglob(name) if p.is_dir())
    hits = [h for h in hits if "testing" not in h.parts] or hits  # prefer training/ (testing has no labels)
    if not hits:
        raise FileNotFoundError(f"no '{name}' folder under {root}")
    return hits[0]


def read_list(path):
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti-root", required=True)
    ap.add_argument("--out", default="datasets/kitti_yolo")
    ap.add_argument("--num-images", type=int, default=3000, help="total images to use (0 = all labelled images)")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--classes", nargs="+", default=["Car", "Pedestrian", "Cyclist"])
    ap.add_argument("--merge", nargs="*", default=[], help="SRC:DST pairs, e.g. Van:Car")
    ap.add_argument("--train-list", default=None, help="optional txt of frame ids (e.g. the 3712/3769 split)")
    ap.add_argument("--val-list", default=None)
    ap.add_argument("--min-box", type=float, default=2.0, help="drop boxes thinner than this (px)")
    ap.add_argument("--link", action="store_true", help="symlink images instead of copying")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.kitti_root)
    img_dir, lab_dir = find_dir(root, "image_2"), find_dir(root, "label_2")
    print(f"images: {img_dir}\nlabels: {lab_dir}")
    cls_map = {c: c for c in args.classes}
    for m in args.merge:
        s, d = m.split(":")
        if d not in args.classes:
            raise ValueError(f"merge target {d} not in --classes")
        cls_map[s] = d
    cls_id = {c: i for i, c in enumerate(args.classes)}

    ids = sorted(p.stem for p in lab_dir.glob("*.txt") if any((img_dir / f"{p.stem}{e}").exists() for e in (".png", ".jpg")))
    print(f"{len(ids)} labelled images found")
    rng = random.Random(args.seed)

    if args.train_list and args.val_list:  # sequence-aware split given by the user
        tr, va = set(read_list(args.train_list)), set(read_list(args.val_list))
        train_ids, val_ids = [i for i in ids if i in tr], [i for i in ids if i in va]
        if args.num_images:
            k_val = round(args.num_images * args.val_ratio)
            train_ids = rng.sample(train_ids, min(len(train_ids), args.num_images - k_val))
            val_ids = rng.sample(val_ids, min(len(val_ids), k_val))
    else:
        if args.num_images:
            ids = rng.sample(ids, min(args.num_images, len(ids)))
        rng.shuffle(ids)
        k_val = round(len(ids) * args.val_ratio)
        val_ids, train_ids = sorted(ids[:k_val]), sorted(ids[k_val:])

    out = Path(args.out)
    stats = {"train": Counter(), "val": Counter()}
    suspicious = 0
    for split, split_ids in (("train", train_ids), ("val", val_ids)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for fid in split_ids:
            src = next(img_dir / f"{fid}{e}" for e in (".png", ".jpg") if (img_dir / f"{fid}{e}").exists())
            im = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            h, w = im.shape[:2]
            lines = []
            for line in (lab_dir / f"{fid}.txt").read_text().splitlines():
                f = line.split()
                if len(f) < 8 or f[0] not in cls_map:
                    continue
                x1, y1, x2, y2 = map(float, f[4:8])
                if x2 > w * 1.05 or y2 > h * 1.05:
                    suspicious += 1
                x1, x2 = max(0.0, x1), min(float(w), x2)
                y1, y2 = max(0.0, y1), min(float(h), y2)
                if x2 - x1 < args.min_box or y2 - y1 < args.min_box:
                    continue
                c = cls_id[cls_map[f[0]]]
                lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                stats[split][args.classes[c]] += 1
            (out / "labels" / split / f"{fid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            dst = out / "images" / split / src.name
            if not dst.exists():
                if args.link:
                    dst.symlink_to(src.resolve())
                else:
                    shutil.copy2(src, dst)

    print(f"\nwritten to {out}")
    for split, ids_ in (("train", train_ids), ("val", val_ids)):
        print(f"  {split}: {len(ids_)} images, boxes {dict(stats[split])}")
    if suspicious:
        print(f"\nWARNING: {suspicious} boxes lie outside their image. If the images were resized to save space,"
              " the KITTI labels (original pixels) no longer match them — rescale the labels or use full-size images.")
    print(f"\nnames for the config: {args.classes}")


if __name__ == "__main__":
    main()
