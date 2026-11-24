"""Build the PointPillars dataset (LiDAR + calib + 3D labels) for the same frames as the 2D YOLO split.

Reads members straight out of the KITTI zips, so the 29 GB Velodyne zip is never fully unpacked:

    python tools/prepare_kitti3d.py \\
        --velo-zip data_object_velodyne.zip --calib-zip data_object_calib.zip --label-zip data_object_label_2.zip \\
        --yolo-root datasets/kitti_yolo --out datasets/kitti3d

--yolo-root: the folder made by prepare_kitti.py; its images/train and images/val decide which frames
             go to train / val (so 2D and 3D experiments use the same frames). Without it, use --num-frames.
Points are cut to the camera field of view (KITTI only labels what the front camera sees),
which also makes the saved point clouds ~5x smaller.
"""

import argparse
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointpillars.kitti import Calib  # noqa: E402


class Source:
    """Reads KITTI files either from a zip or from an extracted folder."""

    def __init__(self, zip_path=None, folder=None, sub=""):
        self.zip = zipfile.ZipFile(zip_path) if zip_path else None
        self.index = {}
        if self.zip:
            for n in self.zip.namelist():
                parts = n.split("/")
                if len(parts) >= 2 and parts[-2] == sub and "testing" not in parts:
                    self.index[Path(n).stem] = n
        else:
            for p in Path(folder).rglob(f"{sub}/*"):
                if "testing" not in p.parts:
                    self.index[p.stem] = p

    def read(self, fid):
        src = self.index[fid]
        return self.zip.read(src) if self.zip else Path(src).read_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velo-zip"); ap.add_argument("--calib-zip"); ap.add_argument("--label-zip")
    ap.add_argument("--kitti-root", help="alternatively: folder with extracted velodyne/, calib/, label_2/")
    ap.add_argument("--yolo-root", default=None)
    ap.add_argument("--num-frames", type=int, default=3000)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--out", default="datasets/kitti3d")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    velo = Source(args.velo_zip, args.kitti_root, "velodyne")
    calib = Source(args.calib_zip, args.kitti_root, "calib")
    label = Source(args.label_zip, args.kitti_root, "label_2")
    print(f"found {len(velo.index)} point clouds, {len(calib.index)} calibs, {len(label.index)} labels")

    sizes = {}
    if args.yolo_root:
        split = {}
        for s in ("train", "val"):
            for p in sorted((Path(args.yolo_root) / "images" / s).glob("*.png")):
                split[p.stem] = s
                im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                sizes[p.stem] = im.shape[:2]
    else:
        ids = sorted(set(velo.index) & set(calib.index) & set(label.index))
        random.Random(args.seed).shuffle(ids)
        ids = ids[: args.num_frames]
        k = round(len(ids) * args.val_ratio)
        split = {i: ("val" if n < k else "train") for n, i in enumerate(ids)}

    out = Path(args.out)
    for d in ("velodyne", "calib", "label_2", "image_size", "ImageSets"):
        (out / d).mkdir(parents=True, exist_ok=True)
    lists = {"train": [], "val": []}
    n_pts_in, n_pts_out, skipped = 0, 0, 0
    for n, (fid, s) in enumerate(sorted(split.items())):
        if fid not in velo.index or fid not in calib.index or fid not in label.index:
            skipped += 1
            continue
        pts = np.frombuffer(velo.read(fid), np.float32).reshape(-1, 4)
        ctext = calib.read(fid).decode()
        (out / "calib" / f"{fid}.txt").write_text(ctext)
        (out / "label_2" / f"{fid}.txt").write_bytes(label.read(fid))
        h, w = sizes.get(fid, (375, 1242))
        (out / "image_size" / f"{fid}.txt").write_text(f"{h} {w}")
        c = Calib(out / "calib" / f"{fid}.txt")
        uv, depth = c.lidar_to_img(pts[:, :3].astype(np.float64))
        fov = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        pts[fov].astype(np.float32).tofile(out / "velodyne" / f"{fid}.bin")
        n_pts_in += len(pts); n_pts_out += int(fov.sum())
        lists[s].append(fid)
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(split)} frames")
    for s, l in lists.items():
        (out / "ImageSets" / f"{s}.txt").write_text("\n".join(l) + "\n")
    print(f"done: train {len(lists['train'])}, val {len(lists['val'])} frames -> {out}"
          f" (kept {n_pts_out / max(n_pts_in, 1):.0%} of points inside the camera view; skipped {skipped})")


if __name__ == "__main__":
    main()
