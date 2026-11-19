"""YOLO26 detection loss.

Per branch:  L = box_gain * CIoU-loss + cls_gain * BCE + l1_gain * (L1 on ltrb distances  | DFL if reg_max > 1)
End-to-end:  L_total = a(t) * L_one2many + (1 - a(t)) * L_one2one          (Progressive Loss, paper Eq. 2)
             a(t) = max(1 - t / max(E - 1, 1), 0) * (a_init - a_final) + a_final   (Eq. 3, a: 0.8 -> 0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assigner import TaskAlignedAssigner
from .utils import bbox2dist, bbox_iou


def progloss_alpha(epoch, epochs, a_init=0.8, a_final=0.1):
    return max(1 - epoch / max(epochs - 1, 1), 0) * (a_init - a_final) + a_final


class BranchLoss(nn.Module):
    def __init__(self, head, assigner, box=7.5, cls=0.5, l1=6.0):
        super().__init__()
        self.head, self.assigner = head, assigner
        self.nc, self.reg_max = head.nc, head.reg_max
        self.gain_box, self.gain_cls, self.gain_l1 = box, cls, l1
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, box_raw, cls_raw, anchor_points, stride_tensor, gt_labels, gt_bboxes, mask_gt):
        dtype = cls_raw.dtype
        pred_bboxes = self.head.decode_boxes(box_raw, anchor_points)  # grid units, (B, A, 4)

        target_bboxes, target_scores, fg_mask = self.assigner(
            cls_raw.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels, gt_bboxes, mask_gt,
        )
        tss = max(target_scores.sum(), 1)

        loss_cls = self.bce(cls_raw, target_scores.to(dtype)).sum() / tss
        loss_box = torch.zeros((), device=cls_raw.device)
        loss_reg = torch.zeros((), device=cls_raw.device)
        if fg_mask.any():
            target_bboxes = target_bboxes / stride_tensor  # -> grid units
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], CIoU=True)
            loss_box = ((1.0 - iou) * weight).sum() / tss

            if self.reg_max > 1:  # DFL (YOLOv8/11 ablation path)
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
                loss_reg = (self._dfl(box_raw[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]) * weight).sum() / tss
            else:  # YOLO26: plain L1 on the four distances
                target_ltrb = bbox2dist(anchor_points, target_bboxes)
                loss_reg = ((box_raw[fg_mask] - target_ltrb[fg_mask]).abs().mean(-1, keepdim=True) * weight).sum() / tss

        items = torch.stack((loss_box * self.gain_box, loss_cls * self.gain_cls, loss_reg * self.gain_l1))
        return items

    @staticmethod
    def _dfl(pred_dist, target):
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        loss = (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr)
        return loss.mean(-1, keepdim=True)


class YOLO26Loss(nn.Module):
    def __init__(self, model, cfg_loss, cfg_assign):
        super().__init__()
        head = model.head
        self.head = head
        self.end2end = head.end2end
        a = cfg_assign
        gains = dict(box=cfg_loss["box"], cls=cfg_loss["cls"], l1=cfg_loss["l1"])
        self.o2m = BranchLoss(head, TaskAlignedAssigner(head.nc, a["topk_o2m"], None, a["alpha"], a["beta"],
                                                        a["stal"], a["stal_smin"], a["stal_sref"]), **gains)
        if self.end2end:
            self.o2o = BranchLoss(head, TaskAlignedAssigner(head.nc, a["topk_o2o"], a["topk2_o2o"], a["alpha"], a["beta"],
                                                            a["stal"], a["stal_smin"], a["stal_sref"]), **gains)
        self.progloss = cfg_loss.get("progloss", True)
        self.a_init, self.a_final = cfg_loss.get("o2m_init", 0.8), cfg_loss.get("o2m_final", 0.1)
        self.alpha = self.a_init if self.progloss else 0.5

    def set_epoch(self, epoch, epochs):
        """Progressive Loss: move weight from the one-to-many to the one-to-one branch once per epoch."""
        self.alpha = progloss_alpha(epoch, epochs, self.a_init, self.a_final) if self.progloss else 0.5

    @staticmethod
    def _targets(targets, bs, device):
        """targets (n, 6) [img_idx, cls, x1, y1, x2, y2] -> padded (B, M, 1), (B, M, 4), (B, M, 1)."""
        if targets.shape[0] == 0:
            z = torch.zeros(bs, 0, 1, device=device)
            return z, torch.zeros(bs, 0, 4, device=device), z
        i = targets[:, 0].long()
        counts = torch.bincount(i, minlength=bs)
        m = int(counts.max())
        out = torch.zeros(bs, m, 5, device=device)
        for j in range(bs):
            sel = targets[i == j]
            if len(sel):
                out[j, : len(sel)] = sel[:, 1:]
        labels, boxes = out[..., :1], out[..., 1:]
        mask = (boxes.sum(-1, keepdim=True) > 0).float()
        return labels, boxes, mask

    def forward(self, preds, targets):
        """Returns (loss for backward, detached items [box, cls, l1, o2m_total, o2o_total])."""
        feats = preds["feats"]
        bs = feats[0].shape[0]
        anchor_points, stride_tensor = self.head.anchors(feats)
        gt_labels, gt_bboxes, mask_gt = self._targets(targets.to(feats[0].device), bs, feats[0].device)

        l_o2m = self.o2m(*preds["one2many"], anchor_points, stride_tensor, gt_labels, gt_bboxes, mask_gt)
        if self.end2end:
            l_o2o = self.o2o(*preds["one2one"], anchor_points, stride_tensor, gt_labels, gt_bboxes, mask_gt)
            items = self.alpha * l_o2m + (1 - self.alpha) * l_o2o
        else:
            l_o2o = torch.zeros_like(l_o2m)
            items = l_o2m
        total = items.sum() * bs  # Ultralytics convention: loss scaled by batch size
        log = torch.cat((items.detach(), l_o2m.sum().detach()[None], l_o2o.sum().detach()[None]))
        return total, log
