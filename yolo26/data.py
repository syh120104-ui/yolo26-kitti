"""YOLO-format dataset with letterbox, Mosaic, random affine, flip and photometric augmentation.

Folder layout (made by tools/prepare_kitti.py):
    <root>/images/{train,val}/*.png
    <root>/labels/{train,val}/*.txt     one line per box: cls cx cy w h   (normalised 0-1)
Input size is rectangular (H, W) because KITTI frames are ~1242x375 (aspect 3.3:1).
"""

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
    def __init__(self, root, split, imgsz=(256, 832), in_ch=1, augment=False, hyp=None):
        self.root = Path(root)
        self.img_files = sorted(p for p in (self.root / "images" / split).iterdir() if p.suffix.lower() in IMG_EXTS)
        if not self.img_files:
            raise FileNotFoundError(f"no images in {self.root / 'images' / split}")
        self.label_files = [self.root / "labels" / split / (p.stem + ".txt") for p in self.img_files]
        self.imgsz = tuple(imgsz)  # (H, W)
        self.in_ch = in_ch
        self.augment = augment
        self.hyp = hyp or {}
        self.mosaic = augment and self.hyp.get("mosaic", 0) > 0
        self.labels = [self._read_label(f) for f in self.label_files]

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def _read_label(path):
        if path.exists() and path.stat().st_size > 0:
            arr = np.loadtxt(path, ndmin=2, dtype=np.float32)
            if arr.size:
                return arr.reshape(-1, 5)
        return np.zeros((0, 5), np.float32)

    # ------------------------------------------------------------------ loading
    def load_image(self, i):
        flag = cv2.IMREAD_GRAYSCALE if self.in_ch == 1 else cv2.IMREAD_COLOR
        img = cv2.imread(str(self.img_files[i]), flag)
        if img is None:
            raise FileNotFoundError(self.img_files[i])
        if img.ndim == 2:
            img = img[..., None]
        h0, w0 = img.shape[:2]
        r = min(self.imgsz[0] / h0, self.imgsz[1] / w0)
        if r != 1:
            img = cv2.resize(img, (round(w0 * r), round(h0 * r)),
                             interpolation=cv2.INTER_LINEAR if r > 1 or self.augment else cv2.INTER_AREA)
            if img.ndim == 2:
                img = img[..., None]
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
        out = np.full((H, W, img.shape[2]), 114, np.uint8)
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
                                         hyp.get("scale", 0.5))
        if self.augment:
            augment_photometric(img, hyp)
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
                canvas = np.full((2 * H, 2 * W, img.shape[2]), 114, np.uint8)
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
                             hyp.get("scale", 0.5))


# ----------------------------------------------------------------------------- augmentation
def random_affine(img, labels, out_hw, degrees=0.0, translate=0.1, scale=0.5):
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
    out = cv2.warpAffine(img, M, dsize=(W, H), borderValue=(114, 114, 114))
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


def augment_photometric(img, hyp):
    """In-place HSV jitter (colour) or brightness/contrast jitter (grayscale)."""
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


def collate_fn(batch):
    imgs, targets, paths = zip(*batch)
    for i, t in enumerate(targets):
        t[:, 0] = i
    return torch.stack(imgs, 0), torch.cat(targets, 0), list(paths)


def build_loader(ds, batch, workers, shuffle):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      collate_fn=collate_fn, drop_last=False, persistent_workers=workers > 0)
