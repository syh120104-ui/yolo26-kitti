"""YOLO-format dataset with letterbox, Mosaic, random affine, flip and photometric augmentation.

Folder layout (made by tools/prepare_kitti.py):
    <root>/images/{train,val}/*.png
    <root>/labels/{train,val}/*.txt     one line per box: cls cx cy w h   (normalised 0-1)
Input size is rectangular (H, W) because KITTI frames are ~1242x375 (aspect 3.3:1).

Optional LiDAR depth channel (made by tools/prepare_lidar.py):
    <root>/depth/{train,val}/<id>.png  uint16, depth in metres * 256, 0 = no LiDAR point
It is stacked after the image channels and goes through the same geometric augmentation.
"""

DEPTH_MAX = 80.0  # metres; depth is encoded as 255 (near) ... 1 (80 m), 0 = no point

import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import xywhn2xyxy

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


class YOLODataset(Dataset):
    def __init__(self, root, split, imgsz=(256, 832), in_ch=1, augment=False, hyp=None, depth=False):
        self.root = Path(root)
        self.img_files = sorted(p for p in (self.root / "images" / split).iterdir() if p.suffix.lower() in IMG_EXTS)
        if not self.img_files:
            raise FileNotFoundError(f"no images in {self.root / 'images' / split}")
        self.label_files = [self.root / "labels" / split / (p.stem + ".txt") for p in self.img_files]
        self.imgsz = tuple(imgsz)  # (H, W)
        self.in_ch = in_ch
        self.depth = depth
        self.img_ch = in_ch - (1 if depth else 0)  # 1 = grayscale, 3 = colour
        assert self.img_ch in (1, 3), "in_channels must be 1/3 (image) or 2/4 (image + LiDAR depth)"
        self.fill = [114] * self.img_ch + [0] * (1 if depth else 0)  # padding value per channel
        if depth:
            self.depth_files = [self.root / "depth" / split / (p.stem + ".png") for p in self.img_files]
            missing = [f for f in self.depth_files if not f.exists()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} depth maps missing, e.g. {missing[0]} — run tools/prepare_lidar.py")
        self.augment = augment
        self.hyp = hyp or {}
        self.mosaic = augment and self.hyp.get("mosaic", 0) > 0
        self.labels = [self._read_label(f) for f in self.label_files]

    def __len__(self):
        return len(self.img_files)

    def mean_box_size(self, per_class=False):
        """Average sqrt(w * h) of the labelled boxes in *network-input* pixels.

        per_class=True returns {class_id: mean size}. Used to set the NWD constant C.
        """
        from PIL import Image

        sizes, classes = [], []
        for f, lab in zip(self.img_files, self.labels):
            if not len(lab):
                continue
            w0, h0 = Image.open(f).size  # header only
            r = min(self.imgsz[0] / h0, self.imgsz[1] / w0)
            sizes.append(np.sqrt(lab[:, 3] * w0 * r * lab[:, 4] * h0 * r))
            classes.append(lab[:, 0].astype(int))
        if not sizes:
            return {} if per_class else 12.8
        sizes, classes = np.concatenate(sizes), np.concatenate(classes)
        if per_class:
            return {int(c): float(sizes[classes == c].mean()) for c in np.unique(classes)}
        return float(sizes.mean())

    @staticmethod
    def _read_label(path):
        if path.exists() and path.stat().st_size > 0:
            arr = np.loadtxt(path, ndmin=2, dtype=np.float32)
            if arr.size:
                return arr.reshape(-1, 5)
        return np.zeros((0, 5), np.float32)

    # ------------------------------------------------------------------ loading
    def load_image(self, i):
        img = read_input(self.img_files[i], self.depth_files[i] if self.depth else None, self.img_ch)
        h0, w0 = img.shape[:2]
        r = min(self.imgsz[0] / h0, self.imgsz[1] / w0)
        if r != 1:
            img = resize_input(img, (round(w0 * r), round(h0 * r)), self.img_ch,
                               cv2.INTER_LINEAR if r > 1 or self.augment else cv2.INTER_AREA)
        return img, (h0, w0)

    def _labels_px(self, i, w, h, padw=0, padh=0):
        lab = self.labels[i].copy()
        if len(lab):
            lab[:, 1:] = xywhn2xyxy(lab[:, 1:], w, h)
            lab[:, [1, 3]] += padw
            lab[:, [2, 4]] += padh
        return lab

    def letterbox(self, img):
        H, W = self.imgsz
        h, w = img.shape[:2]
        top, left = (H - h) // 2, (W - w) // 2
        out = np.empty((H, W, img.shape[2]), np.uint8)
        out[:] = self.fill
        out[top : top + h, left : left + w] = img
        return out, left, top

    # ------------------------------------------------------------------ getitem
    def __getitem__(self, i):
        hyp = self.hyp
        if self.mosaic and random.random() < hyp.get("mosaic", 1.0):
            img, lab = self.load_mosaic(i)
        else:
            im, _ = self.load_image(i)
            h, w = im.shape[:2]
            img, padw, padh = self.letterbox(im)
            lab = self._labels_px(i, w, h, padw, padh)
            if self.augment:
                img, lab = random_affine(img, lab, self.imgsz, hyp.get("degrees", 0), hyp.get("translate", 0.1),
                                         hyp.get("scale", 0.5), self.fill)
        if self.augment:
            augment_photometric(img, hyp, self.img_ch)  # image channels only, never the depth channel
            if random.random() < hyp.get("fliplr", 0.5):
                img = np.ascontiguousarray(img[:, ::-1])
                if len(lab):
                    x1 = self.imgsz[1] - lab[:, 3]
                    x2 = self.imgsz[1] - lab[:, 1]
                    lab[:, 1], lab[:, 3] = x1, x2

        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))  # (C, H, W) uint8
        targets = torch.zeros((len(lab), 6))
        if len(lab):
            targets[:, 1:] = torch.from_numpy(lab)
        return img, targets, str(self.img_files[i])

    def load_mosaic(self, index):
        """4 images on a (2H, 2W) canvas around a random centre, then an affine back to (H, W) (YOLOv4/v5)."""
        H, W = self.imgsz
        yc, xc = int(random.uniform(0.5 * H, 1.5 * H)), int(random.uniform(0.5 * W, 1.5 * W))
        indices = [index] + random.choices(range(len(self)), k=3)
        canvas = None
        all_lab = []
        for k, i in enumerate(indices):
            img, _ = self.load_image(i)
            h, w = img.shape[:2]
            if canvas is None:
                canvas = np.empty((2 * H, 2 * W, img.shape[2]), np.uint8)
                canvas[:] = self.fill
            if k == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif k == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * W), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif k == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * H, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * W), min(2 * H, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            all_lab.append(self._labels_px(i, w, h, x1a - x1b, y1a - y1b))
        lab = np.concatenate(all_lab, 0)
        if len(lab):
            lab[:, 1:] = lab[:, 1:].clip(0, [2 * W, 2 * H, 2 * W, 2 * H])
        hyp = self.hyp
        return random_affine(canvas, lab, self.imgsz, hyp.get("degrees", 0), hyp.get("translate", 0.1),
                             hyp.get("scale", 0.5), self.fill)


# ----------------------------------------------------------------------------- augmentation
def random_affine(img, labels, out_hw, degrees=0.0, translate=0.1, scale=0.5, fill=(114, 114, 114)):
    """Rotate/scale about the image centre, then place it at the output centre +/- translate."""
    H, W = out_hw
    h, w = img.shape[:2]
    C = np.eye(3)
    C[0, 2], C[1, 2] = -w / 2, -h / 2
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * W
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * H
    M = (T @ R @ C)[:2]
    if img.shape[2] <= 4:
        out = cv2.warpAffine(img, M, dsize=(W, H), borderValue=tuple(fill) + (0,) * (4 - len(fill)))
    else:  # cv2 handles at most 4 border values
        out = np.dstack([cv2.warpAffine(img[..., c], M, dsize=(W, H), borderValue=fill[c]) for c in range(img.shape[2])])
    if out.ndim == 2:
        out = out[..., None]

    if len(labels):
        n = len(labels)
        xy = np.ones((n * 4, 3))
        xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = (xy @ M.T).reshape(n, 8)
        x, y = xy[:, [0, 2, 4, 6]], xy[:, [1, 3, 5, 7]]
        new = np.stack((x.min(1), y.min(1), x.max(1), y.max(1)), 1)
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, W)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, H)
        keep = box_candidates(labels[:, 1:5].T * s, new.T)
        labels = labels[keep].copy()
        labels[:, 1:5] = new[keep]
    return out, labels.astype(np.float32)


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def augment_photometric(img, hyp, img_ch=None):
    """In-place HSV jitter (colour) or brightness/contrast jitter (grayscale) on the first img_ch channels."""
    img_ch = img_ch or img.shape[2]
    if img_ch != img.shape[2]:  # image + depth: jitter a copy of the image channels, write it back
        part = np.ascontiguousarray(img[..., :img_ch])
        augment_photometric(part, hyp, img_ch)
        img[..., :img_ch] = part
        return
    hv = hyp.get("hsv_v", 0.4)
    if img.shape[2] == 3:
        hh, hs = hyp.get("hsv_h", 0.015), hyp.get("hsv_s", 0.7)
        r = np.random.uniform(-1, 1, 3) * [hh, hs, hv] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        x = np.arange(0, 256, dtype=r.dtype)
        lut_h = ((x * r[0]) % 180).astype(np.uint8)
        lut_s = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_v = np.clip(x * r[2], 0, 255).astype(np.uint8)
        hsv = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=img)
    else:
        gain = 1 + random.uniform(-hv, hv)  # brightness
        contrast = 1 + random.uniform(-hv / 2, hv / 2)
        mean = img.mean()
        x = (img.astype(np.float32) - mean) * contrast + mean * gain
        img[...] = np.clip(x, 0, 255).astype(np.uint8)


def read_input(img_path, depth_path=None, img_ch=1):
    """Original-size (h, w, C) uint8 array: image channels (+ one LiDAR depth channel)."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE if img_ch == 1 else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)
    if img.ndim == 2:
        img = img[..., None]
    if depth_path is None:
        return img
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(depth_path)
    return np.concatenate((img, encode_depth(d)[..., None]), 2)


def encode_depth(d16):
    """uint16 (metres*256, 0 = empty) -> uint8 inverse-ish depth: near = bright, far = dark, empty = 0."""
    m = d16.astype(np.float32) / 256.0
    out = np.zeros(m.shape, np.uint8)
    valid = m > 0
    out[valid] = np.clip(255.0 * (1.0 - np.minimum(m[valid], DEPTH_MAX) / DEPTH_MAX), 1, 255).astype(np.uint8)
    return out


def resize_input(img, size_wh, img_ch, interp):
    """Image channels with interp; the sparse depth channel with nearest (averaging would mix in empty pixels)."""
    a = cv2.resize(img[..., :img_ch], size_wh, interpolation=interp)
    if a.ndim == 2:
        a = a[..., None]
    if img.shape[2] == img_ch:
        return a
    d = cv2.resize(img[..., img_ch:], size_wh, interpolation=cv2.INTER_NEAREST)
    if d.ndim == 2:
        d = d[..., None]
    return np.concatenate((a, d), 2)


def collate_fn(batch):
    imgs, targets, paths = zip(*batch)
    for i, t in enumerate(targets):
        t[:, 0] = i
    return torch.stack(imgs, 0), torch.cat(targets, 0), list(paths)


def build_loader(ds, batch, workers, shuffle):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      collate_fn=collate_fn, drop_last=False, persistent_workers=workers > 0)
