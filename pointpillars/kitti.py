"""KITTI LiDAR data: calibration, 3D labels, point clouds, augmentation and pillar creation.

Folder layout (made by tools/prepare_kitti3d.py):
    <root>/velodyne/<id>.bin     float32 (N, 4): x, y, z, reflectance — already cut to camera FOV + range
    <root>/calib/<id>.txt        KITTI calibration
    <root>/label_2/<id>.txt      KITTI labels (3D fields: h w l, x y z in camera coords, rotation_y)
    <root>/image_size/<id>.txt   "height width" of the left colour image (for 2D box heights / difficulty)
    <root>/ImageSets/{train,val}.txt
"""

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ----------------------------------------------------------------------------- calibration
class Calib:
    def __init__(self, path):
        vals = {}
        for line in Path(path).read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                vals[k.strip()] = np.array([float(x) for x in v.split()], np.float64)
        self.P2 = vals["P2"].reshape(3, 4)
        self.R0 = vals["R0_rect"].reshape(3, 3)
        tr = vals.get("Tr_velo_to_cam", vals.get("Tr_velo_cam")).reshape(3, 4)
        self.V2C = np.vstack((tr, [0, 0, 0, 1]))  # 4x4

    def lidar_to_rect(self, pts):
        """(N, 3) LiDAR -> (N, 3) rectified camera coordinates."""
        hom = np.hstack((pts[:, :3], np.ones((len(pts), 1))))
        return (self.R0 @ (self.V2C @ hom.T)[:3]).T

    def rect_to_lidar(self, pts):
        ref = np.linalg.inv(self.R0) @ pts.T
        hom = np.vstack((ref, np.ones((1, len(pts)))))
        return (np.linalg.inv(self.V2C) @ hom)[:3].T

    def rect_to_img(self, pts):
        """(N, 3) rect -> (N, 2) pixels and (N,) depth."""
        hom = np.hstack((pts, np.ones((len(pts), 1))))
        p = (self.P2 @ hom.T).T
        return p[:, :2] / p[:, 2:3], p[:, 2]

    def lidar_to_img(self, pts):
        return self.rect_to_img(self.lidar_to_rect(pts))


# ----------------------------------------------------------------------------- labels
def read_labels(path, calib, classes, ignore_map=None):
    """Return dict with LiDAR boxes (N, 7), class ids, names, 2D bbox heights, occlusion, truncation.

    Objects of other classes are kept with cls = -1 (e.g. Van, DontCare) so evaluation can ignore them.
    """
    names, boxes, h2d, occ, trunc = [], [], [], [], []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) < 15:
            continue
        names.append(f[0])
        trunc.append(float(f[1]))
        occ.append(int(float(f[2])))
        h2d.append(float(f[7]) - float(f[5]))
        h, w, l = map(float, f[8:11])
        loc = np.array([[float(f[11]), float(f[12]), float(f[13])]])
        ry = float(f[14])
        xyz = calib.rect_to_lidar(loc)[0]
        xyz[2] += h / 2  # KITTI location = bottom centre -> box centre
        boxes.append([xyz[0], xyz[1], xyz[2], l, w, h, -(math.pi / 2 + ry)])
    cls = np.array([classes.index(n) if n in classes else -1 for n in names], np.int64)
    return {
        "boxes": np.array(boxes, np.float32).reshape(-1, 7),
        "cls": cls,
        "names": np.array(names),
        "h2d": np.array(h2d, np.float32),
        "occ": np.array(occ, np.int64),
        "trunc": np.array(trunc, np.float32),
    }


# ----------------------------------------------------------------------------- pillars
def make_pillars(points, pc_range, voxel_size, max_points, max_pillars, shuffle=False):
    """Group points into vertical pillars and decorate them (PointPillars Sec. 2.1, D = 9 features).

    Returns features (P, N, 9), coords (P, 2) as (row=y index, col=x index), point mask (P, N).
    """
    x0, y0, z0, x1, y1, z1 = pc_range
    keep = ((points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)
            & (points[:, 2] >= z0) & (points[:, 2] < z1))
    pts = points[keep]
    if shuffle:
        pts = pts[np.random.permutation(len(pts))]
    nx = int(round((x1 - x0) / voxel_size[0]))
    ix = ((pts[:, 0] - x0) / voxel_size[0]).astype(np.int64)
    iy = ((pts[:, 1] - y0) / voxel_size[1]).astype(np.int64)
    key = iy * nx + ix
    order = np.argsort(key, kind="stable")
    key, pts, ix, iy = key[order], pts[order], ix[order], iy[order]
    uniq_all, start_all, count_all = np.unique(key, return_index=True, return_counts=True)
    rank = np.arange(len(key)) - np.repeat(start_all, count_all)  # index of each point inside its pillar
    uniq = uniq_all
    if len(uniq) > max_pillars:  # keep a subset of pillars (random while training)
        sel = np.sort(np.random.choice(len(uniq), max_pillars, replace=False)) if shuffle else np.arange(max_pillars)
        uniq = uniq[sel]
    P = len(uniq)
    if P == 0:
        return (np.zeros((0, max_points, 9), np.float32), np.zeros((0, 2), np.int64),
                np.zeros((0, max_points), bool))
    pillar_of = np.full(len(key), -1)
    pos = np.searchsorted(uniq, key)
    valid = (pos < P) & (uniq[np.minimum(pos, P - 1)] == key) & (rank < max_points)
    pillar_of[valid] = pos[valid]

    raw = np.zeros((P, max_points, 4), np.float32)
    mask = np.zeros((P, max_points), bool)
    raw[pillar_of[valid], rank[valid]] = pts[valid, :4]
    mask[pillar_of[valid], rank[valid]] = True

    n = np.maximum(mask.sum(1, keepdims=True), 1)
    mean = raw[..., :3].sum(1) / n  # (P, 3) mean of the pillar's points
    coords = np.stack((uniq // nx, uniq % nx), 1)  # (row, col)
    cx = coords[:, 1] * voxel_size[0] + voxel_size[0] / 2 + x0
    cy = coords[:, 0] * voxel_size[1] + voxel_size[1] / 2 + y0
    feats = np.concatenate((
        raw,                                            # x, y, z, r
        raw[..., :3] - mean[:, None],                   # x_c, y_c, z_c  (offset to the pillar mean)
        raw[..., :1] - cx[:, None, None],               # x_p           (offset to the pillar centre)
        raw[..., 1:2] - cy[:, None, None],              # y_p
    ), -1).astype(np.float32)
    feats *= mask[..., None]
    return feats, coords.astype(np.int64), mask


# ----------------------------------------------------------------------------- dataset
class KittiLidar(Dataset):
    def __init__(self, root, split, cfg, augment=False):
        self.root = Path(root)
        self.ids = (self.root / "ImageSets" / f"{split}.txt").read_text().split()
        self.cfg = cfg
        self.classes = cfg["classes"]
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def load(self, fid):
        pts = np.fromfile(self.root / "velodyne" / f"{fid}.bin", np.float32).reshape(-1, 4)
        calib = Calib(self.root / "calib" / f"{fid}.txt")
        lab = read_labels(self.root / "label_2" / f"{fid}.txt", calib, self.classes)
        return pts, calib, lab

    def __getitem__(self, i):
        fid = self.ids[i]
        pts, _, lab = self.load(fid)
        keep = lab["cls"] >= 0
        boxes, cls = lab["boxes"][keep].copy(), lab["cls"][keep].copy()
        if self.augment:
            pts, boxes = augment(pts, boxes, self.cfg["augment"])
        feats, coords, mask = make_pillars(pts, self.cfg["pc_range"], self.cfg["voxel_size"],
                                           self.cfg["max_points"], self.cfg["max_pillars_train" if self.augment
                                                                            else "max_pillars_test"],
                                           shuffle=self.augment)
        return {"feats": torch.from_numpy(feats), "coords": torch.from_numpy(coords),
                "boxes": torch.from_numpy(boxes), "cls": torch.from_numpy(cls), "id": fid}


def augment(pts, boxes, a):
    """Global augmentation (PointPillars Sec. 3.3): random flip over the x axis, rotation, scaling."""
    pts, boxes = pts.copy(), boxes.copy()
    if np.random.rand() < a.get("flip", 0.5):  # mirror left/right: y -> -y
        pts[:, 1] *= -1
        boxes[:, 1] *= -1
        boxes[:, 6] *= -1
    rot = np.random.uniform(*a.get("rotation", [-math.pi / 4, math.pi / 4]))
    c, s = math.cos(rot), math.sin(rot)
    R = np.array([[c, -s], [s, c]], np.float32)
    pts[:, :2] = pts[:, :2] @ R.T
    boxes[:, :2] = boxes[:, :2] @ R.T
    boxes[:, 6] += rot
    sc = np.random.uniform(*a.get("scale", [0.95, 1.05]))
    pts[:, :3] *= sc
    boxes[:, :6] *= sc
    return pts, boxes


def collate(batch):
    feats, coords, boxes, cls, ids = [], [], [], [], []
    for b, d in enumerate(batch):
        feats.append(d["feats"])
        coords.append(torch.cat((torch.full((len(d["coords"]), 1), b, dtype=torch.long), d["coords"]), 1))
        boxes.append(d["boxes"])
        cls.append(d["cls"])
        ids.append(d["id"])
    return {"feats": torch.cat(feats), "coords": torch.cat(coords), "boxes": boxes, "cls": cls, "ids": ids,
            "batch_size": len(batch)}


def build_loader(ds, batch, workers, shuffle):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
