"""YOLO26 detector: YOLO11-style backbone/neck + DFL-free dual (one-to-many / one-to-one) head.

Layer list follows Fig. S1 of "Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models"
(Jocher et al., arXiv:2606.03748). Scale multipliers are those of the YOLO11 family.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .nn import C2PSA, SPPF, C3k2, C3k2BoT, Conv, DWConv
from .utils import dist2bbox, make_anchors, make_divisible

# depth multiple, width multiple, max channels  (same table as YOLO11)
SCALES = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}


class DFL(nn.Module):
    """Distribution Focal Loss integral: softmax over reg_max bins -> expected distance (YOLOv8/11)."""

    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        self.register_buffer("proj", torch.arange(c1, dtype=torch.float))

    def forward(self, x):  # x: (B, 4*c1, A)
        b, _, a = x.shape
        return (x.view(b, 4, self.c1, a).softmax(2) * self.proj.view(1, 1, -1, 1)).sum(2)


class Detect26(nn.Module):
    """Decoupled anchor-free head.

    reg_max=1  -> YOLO26: box branch outputs 4 raw ltrb distances (no DFL), trained with IoU + L1.
    reg_max>1  -> YOLO11/v8 style DFL head (kept only for ablation).
    end2end    -> an extra one-to-one copy of both branches (YOLOv10 / YOLO26). Inference uses it, no NMS.
    """

    def __init__(self, nc, ch, strides, reg_max=1, end2end=True, o2o_detach=True, max_det=300, ref_ch=None):
        super().__init__()
        self.nc, self.nl, self.reg_max = nc, len(ch), reg_max
        self.no = nc + 4 * reg_max
        self.end2end, self.o2o_detach, self.max_det = end2end, o2o_detach, max_det
        self.register_buffer("stride", torch.tensor(strides, dtype=torch.float))

        ref = ref_ch or ch[0]  # with a P2 head, keep the branch widths the P3-first model had
        c2 = max(16, ref // 4, reg_max * 4)
        c3 = max(ref, min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()
        if end2end:
            self.one2one_cv2 = deepcopy(self.cv2)
            self.one2one_cv3 = deepcopy(self.cv3)
        self._anchor_cache = (None, None, None)

    def bias_init(self, img_hw=(640, 640)):
        """Start every box at ~1 cell and every class at a low prior probability (YOLOv5 practice)."""
        area = img_hw[0] * img_hw[1]
        pairs = [(self.cv2, self.cv3)] + ([(self.one2one_cv2, self.one2one_cv3)] if self.end2end else [])
        for box_branch, cls_branch in pairs:
            for a, b, s in zip(box_branch, cls_branch, self.stride.tolist()):
                a[-1].bias.data[:] = 1.0
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (area / s**2))

    @staticmethod
    def _run(feats, box_branch, cls_branch):
        b = feats[0].shape[0]
        box = torch.cat([box_branch[i](f).view(b, -1, f.shape[2] * f.shape[3]) for i, f in enumerate(feats)], 2)
        cls = torch.cat([cls_branch[i](f).view(b, -1, f.shape[2] * f.shape[3]) for i, f in enumerate(feats)], 2)
        return box.permute(0, 2, 1).contiguous(), cls.permute(0, 2, 1).contiguous()  # (B,A,4*reg_max), (B,A,nc)

    def forward(self, feats):
        out = {"feats": feats, "one2many": self._run(feats, self.cv2, self.cv3)}
        if self.end2end:
            src = [f.detach() for f in feats] if self.o2o_detach else feats
            out["one2one"] = self._run(src, self.one2one_cv2, self.one2one_cv3)
        return out

    # ------------------------------------------------------------------ decoding
    def anchors(self, feats):
        shapes = tuple(f.shape[2:] for f in feats)
        if self._anchor_cache[0] != shapes or self._anchor_cache[1].device != feats[0].device:
            ap, st = make_anchors(feats, self.stride.tolist())
            self._anchor_cache = (shapes, ap, st)
        return self._anchor_cache[1], self._anchor_cache[2]

    def decode_boxes(self, box, anchor_points):
        """(B,A,4*reg_max) raw -> (B,A,4) xyxy in grid units."""
        if self.reg_max > 1:
            box = self.dfl(box.permute(0, 2, 1)).permute(0, 2, 1)
        return dist2bbox(box, anchor_points.unsqueeze(0))

    @torch.no_grad()
    def predict(self, out, branch="one2one", conf=0.001, iou=0.7, max_det=None):
        """Turn raw head output into a list of (n, 6) tensors [x1, y1, x2, y2, score, cls] in input pixels.

        branch="one2one": NMS-free top-k (YOLO26 default end-to-end inference)
        branch="one2many": classic path with class-aware NMS
        """
        max_det = max_det or self.max_det
        box, cls = out[branch]
        ap, st = self.anchors(out["feats"])
        boxes = self.decode_boxes(box.float(), ap) * st.unsqueeze(0)  # (B,A,4) pixels
        scores = cls.float().sigmoid()  # (B,A,nc)
        results = []
        for bx, sc in zip(boxes, scores):
            if branch == "one2one":
                # top-k anchors by best class score, then top-k over (anchor, class) pairs — no NMS needed
                k = min(max_det, sc.shape[0])
                top_anchor = sc.amax(1).topk(k).indices
                bx, sc = bx[top_anchor], sc[top_anchor]
                s, idx = sc.flatten().topk(k)
                det = torch.cat((bx[idx // self.nc], s[:, None], (idx % self.nc)[:, None].float()), 1)
                det = det[det[:, 4] > conf]
            else:
                s, c = sc.max(1)
                keep = s > conf
                bx, s, c = bx[keep], s[keep], c[keep]
                keep = torchvision.ops.batched_nms(bx, s, c, iou)[:max_det]
                det = torch.cat((bx[keep], s[keep, None], c[keep, None].float()), 1)
            results.append(det)
        return results


class YOLO26(nn.Module):
    def __init__(self, nc=3, in_ch=1, scale="n", reg_max=1, end2end=True, o2o_detach=True, max_det=300,
                 botnet=False, botnet_heads=4, imgsz=(256, 832), p2=False):
        super().__init__()
        depth, width, max_ch = SCALES[scale]
        c = lambda x: make_divisible(min(x, max_ch) * width, 8)  # noqa: E731
        d = lambda n: max(round(n * depth), 1) if n > 1 else n  # noqa: E731
        big = scale in "mlx"  # YOLO11/26 rule: C3k inner blocks everywhere for m/l/x
        self.yaml = dict(nc=nc, in_ch=in_ch, scale=scale, reg_max=reg_max, end2end=end2end,
                         o2o_detach=o2o_detach, max_det=max_det, botnet=botnet, p2=p2)

        # ---------------- backbone
        self.b0 = Conv(in_ch, c(64), 3, 2)  # P1/2
        self.b1 = Conv(c(64), c(128), 3, 2)  # P2/4
        self.b2 = C3k2(c(128), c(256), d(2), c3k=big, e=0.25)
        self.b3 = Conv(c(256), c(256), 3, 2)  # P3/8
        self.b4 = C3k2(c(256), c(512), d(2), c3k=big, e=0.25)
        self.b5 = Conv(c(512), c(512), 3, 2)  # P4/16
        self.b6 = C3k2(c(512), c(512), d(2), c3k=True)
        self.b7 = Conv(c(512), c(1024), 3, 2)  # P5/32
        if botnet:  # BoTNet: last backbone stage (P5) uses MHSA bottlenecks instead of 3x3-conv bottlenecks
            self.b8 = C3k2BoT(c(1024), c(1024), d(2), heads=botnet_heads, ref_hw=(imgsz[0] // 32, imgsz[1] // 32))
        else:
            self.b8 = C3k2(c(1024), c(1024), d(2), c3k=True)
        self.b9 = SPPF(c(1024), c(1024), 5)
        self.b10 = C2PSA(c(1024), c(1024), d(2))

        # ---------------- neck (top-down FPN + bottom-up PAN)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.n13 = C3k2(c(1024) + c(512), c(512), d(2), c3k=big)
        self.n16 = C3k2(c(512) + c(512), c(256), d(2), c3k=big)  # P3 out
        self.n17 = Conv(c(256), c(256), 3, 2)
        self.n19 = C3k2(c(256) + c(512), c(512), d(2), c3k=big)  # P4 out
        self.n20 = Conv(c(512), c(512), 3, 2)
        self.n22 = C3k2(c(512) + c(1024), c(1024), d(1), c3k=True, e=0.5, attn=True)  # P5 out

        self.p2 = p2
        if p2:  # extra stride-4 level (YOLOv8-p2 style): top-down P3 -> P2, then bottom-up back into P3
            self.np2 = C3k2(c(256) + c(256), c(128), d(2), c3k=big)  # P2 out
            self.np2_down = Conv(c(128), c(128), 3, 2)
            self.np3 = C3k2(c(128) + c(256), c(256), d(2), c3k=big)  # P3 out (refined with P2 detail)
            self.head = Detect26(nc, (c(128), c(256), c(512), c(1024)), (4, 8, 16, 32), reg_max, end2end, o2o_detach,
                                 max_det, ref_ch=c(256))
        else:
            self.head = Detect26(nc, (c(256), c(512), c(1024)), (8, 16, 32), reg_max, end2end, o2o_detach, max_det)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps, m.momentum = 1e-3, 0.03
        self.head.bias_init()

    def forward(self, x):
        p2 = self.b2(self.b1(self.b0(x)))
        p3 = self.b4(self.b3(p2))
        p4 = self.b6(self.b5(p3))
        p5 = self.b10(self.b9(self.b8(self.b7(p4))))

        n13 = self.n13(torch.cat((self.up(p5), p4), 1))
        n16 = self.n16(torch.cat((self.up(n13), p3), 1))
        if self.p2:
            o2 = self.np2(torch.cat((self.up(n16), p2), 1))
            n16 = self.np3(torch.cat((self.np2_down(o2), n16), 1))
        n19 = self.n19(torch.cat((self.n17(n16), n13), 1))
        n22 = self.n22(torch.cat((self.n20(n19), p5), 1))
        return self.head([o2, n16, n19, n22] if self.p2 else [n16, n19, n22])


def build_model(cfg):
    """Create YOLO26 from a config dict (keeps train / val / tools in sync)."""
    m = cfg["model"]
    return YOLO26(len(cfg["data"]["names"]), m["in_channels"], m["scale"], m["reg_max"], m["end2end"],
                  m["o2o_detach"], m["max_det"], m.get("botnet", False), m.get("botnet_heads", 4),
                  tuple(cfg["train"]["imgsz"]), m.get("p2", False))


def model_info(model: nn.Module, imgsz=(256, 832), in_ch=1):
    n_p = sum(p.numel() for p in model.parameters())
    gflops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode

        x = torch.zeros(1, in_ch, *imgsz, device=next(model.parameters()).device)
        model.eval()
        with FlopCounterMode(display=False) as fc:
            model(x)
        gflops = fc.get_total_flops() / 1e9  # = 2 x MACs, the convention YOLO papers report
    except Exception:  # noqa: BLE001
        pass
    return n_p, gflops


if __name__ == "__main__":
    # Sanity check against the paper (Table 7, COCO 80 classes, 640x640, one head at inference):
    # n 2.4M/5.4G, s 9.5M/20.7G, m 20.4M/68.2G, l 24.8M/86.4G, x 55.7M/193.9G
    for s in "nsmlx":
        m = YOLO26(nc=80, in_ch=3, scale=s, end2end=False)  # count a single head, as deployed
        p, g = model_info(m, (640, 640), 3)
        print(f"YOLO26{s}: {p / 1e6:.2f}M params, {g:.1f} GFLOPs @640 (paper Table 7 for comparison)")
