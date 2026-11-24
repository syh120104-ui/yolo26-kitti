"""PointPillars (Lang et al., CVPR 2019): Pillar Feature Net -> pseudo-image -> 2D backbone -> SSD head.

Layer sizes follow the paper / the common OpenPCDet KITTI config:
    PFN 9 -> 64, backbone blocks (3, 5, 5) layers, strides (2, 2, 2), widths (64, 128, 256),
    upsampled to stride 2 with 128 channels each -> 384-channel map, anchor head (2 rotations per class).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .box_ops import bev_aabb, decode_boxes, encode_boxes, limit_period, nearest_bev_iou


class PillarFeatureNet(nn.Module):
    """Simplified PointNet: Linear + BatchNorm + ReLU per point, then max over the points of a pillar."""

    def __init__(self, c_in=9, c_out=64):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out, bias=False)
        self.bn = nn.BatchNorm1d(c_out, eps=1e-3, momentum=0.01)

    def forward(self, feats):  # (P, N, 9)
        x = self.linear(feats)
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        return F.relu(x).max(1).values  # (P, 64)


def scatter(pillar_feats, coords, batch_size, grid_hw):
    """Put each pillar vector back at its (row, col) cell -> (B, C, H, W) pseudo-image (paper Sec. 2.1)."""
    H, W = grid_hw
    canvas = pillar_feats.new_zeros(batch_size * H * W, pillar_feats.shape[1])
    idx = coords[:, 0] * H * W + coords[:, 1] * W + coords[:, 2]
    canvas[idx] = pillar_feats
    return canvas.view(batch_size, H, W, -1).permute(0, 3, 1, 2).contiguous()


def conv_bn(c1, c2, s=1):
    return nn.Sequential(nn.Conv2d(c1, c2, 3, s, 1, bias=False), nn.BatchNorm2d(c2, eps=1e-3, momentum=0.01),
                         nn.ReLU(inplace=True))


class Backbone(nn.Module):
    """Top-down blocks + transposed-conv upsampling and concatenation (paper Fig. 2)."""

    def __init__(self, c_in=64, layers=(3, 5, 5), strides=(2, 2, 2), widths=(64, 128, 256),
                 up_strides=(1, 2, 4), up_width=128):
        super().__init__()
        self.blocks, self.ups = nn.ModuleList(), nn.ModuleList()
        c = c_in
        for n, s, w, us in zip(layers, strides, widths, up_strides):
            self.blocks.append(nn.Sequential(conv_bn(c, w, s), *[conv_bn(w, w) for _ in range(n)]))
            self.ups.append(nn.Sequential(nn.ConvTranspose2d(w, up_width, us, us, bias=False),
                                          nn.BatchNorm2d(up_width, eps=1e-3, momentum=0.01), nn.ReLU(inplace=True)))
            c = w
        self.out_channels = up_width * len(layers)

    def forward(self, x):
        outs = []
        for blk, up in zip(self.blocks, self.ups):
            x = blk(x)
            outs.append(up(x))
        return torch.cat(outs, 1)


class PointPillars(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.classes = cfg["classes"]
        self.nc = len(self.classes)
        x0, y0, _, x1, y1, _ = cfg["pc_range"]
        vx, vy = cfg["voxel_size"][:2]
        self.grid_hw = (int(round((y1 - y0) / vy)), int(round((x1 - x0) / vx)))  # (rows=y, cols=x)
        self.pfn = PillarFeatureNet(9, 64)
        self.backbone = Backbone(64)
        self.out_stride = 2
        self.rots = cfg["anchor_rotations"]
        self.na = self.nc * len(self.rots)  # anchors per location
        C = self.backbone.out_channels
        self.cls_head = nn.Conv2d(C, self.na * self.nc, 1)
        self.box_head = nn.Conv2d(C, self.na * 7, 1)
        self.dir_head = nn.Conv2d(C, self.na * 2, 1)
        nn.init.constant_(self.cls_head.bias, -math.log((1 - 0.01) / 0.01))
        nn.init.normal_(self.box_head.weight, std=0.001)
        nn.init.zeros_(self.box_head.bias)
        self._anchors = None

    # ------------------------------------------------------------------ anchors
    def anchors(self, device):
        """(H', W', na, 7) anchors on the stride-2 feature map; order = class-major, then rotation."""
        if self._anchors is not None and self._anchors.device == device:
            return self._anchors
        x0, y0, _, x1, y1, _ = self.cfg["pc_range"]
        H, W = self.grid_hw[0] // self.out_stride, self.grid_hw[1] // self.out_stride
        xs = torch.linspace(x0, x1, W + 1, device=device)[:-1] + (x1 - x0) / W / 2
        ys = torch.linspace(y0, y1, H + 1, device=device)[:-1] + (y1 - y0) / H / 2
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        per = []
        for name in self.classes:
            a = self.cfg["anchors"][name]  # size [dx, dy, dz], z centre
            for r in self.rots:
                per.append(torch.tensor([*a["size"], r], device=device))
        sizes = torch.stack(per)  # (na, 4): dx, dy, dz, rot
        zc = torch.tensor([self.cfg["anchors"][n]["z"] for n in self.classes for _ in self.rots], device=device)
        anc = torch.zeros(H, W, self.na, 7, device=device)
        anc[..., 0], anc[..., 1] = xx[..., None], yy[..., None]
        anc[..., 2] = zc
        anc[..., 3:6] = sizes[:, :3]
        anc[..., 6] = sizes[:, 3]
        self._anchors = anc
        return anc

    def anchor_class(self, device):
        return torch.arange(self.nc, device=device).repeat_interleave(len(self.rots))  # (na,)

    # ------------------------------------------------------------------ forward
    def forward(self, batch):
        x = self.pfn(batch["feats"])
        x = scatter(x, batch["coords"], batch["batch_size"], self.grid_hw)
        x = self.backbone(x)
        B, _, H, W = x.shape
        cls = self.cls_head(x).view(B, self.na, self.nc, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, self.nc)
        box = self.box_head(x).view(B, self.na, 7, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, 7)
        dirc = self.dir_head(x).view(B, self.na, 2, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1, 2)
        return cls, box, dirc

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, out, score_thr=0.1, nms_iou=0.01, max_det=100):
        """Decode, keep the best class per anchor, axis-aligned BEV NMS per class (paper Sec. 3.2)."""
        cls, box, dirc = out
        anchors = self.anchors(cls.device).view(-1, 7)
        dir_off = self.cfg["dir_offset"]
        results = []
        for b in range(cls.shape[0]):
            scores, labels = cls[b].sigmoid().max(1)
            keep = scores > score_thr
            if keep.sum() > 4096:
                keep = keep & (scores >= scores[keep].topk(4096).values[-1])
            boxes = decode_boxes(box[b][keep], anchors[keep])
            dlab = dirc[b][keep].argmax(1)
            rot = limit_period(boxes[:, 6] - dir_off, 0.0, math.pi)
            boxes[:, 6] = rot + dir_off + math.pi * dlab.float()
            boxes[:, 6] = limit_period(boxes[:, 6], 0.5, 2 * math.pi)
            s, l = scores[keep], labels[keep]
            k = torchvision.ops.batched_nms(bev_aabb(boxes), s, l, nms_iou)[:max_det]
            results.append((boxes[k], s[k], l[k]))
        return results


# ----------------------------------------------------------------------------- targets + loss
class PillarLoss(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()
        self.m, self.cfg = model, cfg
        self.w_loc, self.w_cls, self.w_dir = cfg["loss"]["loc"], cfg["loss"]["cls"], cfg["loss"]["dir"]

    @torch.no_grad()
    def assign(self, gt_boxes, gt_cls, device):
        """Per class: IoU >= match -> positive, < unmatched -> negative, else ignored; each GT keeps its best anchor."""
        anchors = self.m.anchors(device).view(-1, 7)
        acls = self.m.anchor_class(device).repeat(anchors.shape[0] // self.m.na)
        A = anchors.shape[0]
        labels = torch.zeros(A, dtype=torch.long, device=device)  # -1 ignore, 0 negative, c+1 positive
        targets = torch.zeros(A, 7, device=device)
        for c, name in enumerate(self.m.classes):
            am = acls == c
            gm = gt_cls == c
            idx_a = am.nonzero().squeeze(1)
            thr_m, thr_u = self.cfg["anchors"][name]["match"], self.cfg["anchors"][name]["unmatch"]
            if gm.sum() == 0:
                continue
            iou = nearest_bev_iou(anchors[idx_a], gt_boxes[gm])  # (Ac, Gc)
            best_iou, best_gt = iou.max(1)
            lab = torch.zeros(len(idx_a), dtype=torch.long, device=device)
            lab[(best_iou >= thr_u) & (best_iou < thr_m)] = -1
            pos = best_iou >= thr_m
            ok = iou.max(0).values > 0
            force = iou.argmax(0)[ok]  # best anchor of every GT is always positive
            pos[force] = True
            best_gt[force] = torch.arange(iou.shape[1], device=device)[ok]
            lab[pos] = c + 1
            labels[idx_a] = lab
            g = gt_boxes[gm][best_gt[pos]]
            targets[idx_a[pos]] = g
        return labels, targets

    def forward(self, out, batch):
        cls, box, dirc = out
        device = cls.device
        anchors = self.m.anchors(device).view(-1, 7)
        l_cls = l_loc = l_dir = cls.new_zeros(())
        n_pos_total = 0
        for b in range(cls.shape[0]):
            gt = batch["boxes"][b].to(device)
            gc = batch["cls"][b].to(device)
            labels, tgt = self.assign(gt, gc, device)
            pos = labels > 0
            valid = labels >= 0
            n_pos = max(int(pos.sum()), 1)
            n_pos_total += int(pos.sum())
            onehot = torch.zeros_like(cls[b])
            onehot[pos, labels[pos] - 1] = 1
            l_cls = l_cls + sigmoid_focal(cls[b][valid], onehot[valid]) / n_pos
            if pos.any():
                enc = encode_boxes(tgt[pos], anchors[pos])
                pred = box[b][pos]
                # sin(a - b) = sin a cos b - cos a sin b : heading residual through the sine (paper Eq. 1)
                p_r, t_r = pred[:, 6:7], enc[:, 6:7]
                pred_s = torch.cat((pred[:, :6], torch.sin(p_r) * torch.cos(t_r)), 1)
                tgt_s = torch.cat((enc[:, :6], torch.cos(p_r) * torch.sin(t_r)), 1)
                l_loc = l_loc + F.smooth_l1_loss(pred_s, tgt_s, beta=1 / 9, reduction="sum") / n_pos
                rot = limit_period(tgt[pos][:, 6] - self.cfg["dir_offset"], 0.0, 2 * math.pi)
                dir_t = torch.clamp((rot / math.pi).floor().long(), 0, 1)
                l_dir = l_dir + F.cross_entropy(dirc[b][pos], dir_t, reduction="sum") / n_pos
        B = cls.shape[0]
        items = torch.stack((self.w_loc * l_loc, self.w_cls * l_cls, self.w_dir * l_dir)) / B
        return items.sum(), items.detach(), n_pos_total


def sigmoid_focal(logits, targets, alpha=0.25, gamma=2.0):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * ce).sum()
