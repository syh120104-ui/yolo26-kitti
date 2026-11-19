"""Task-Aligned label assignment (TAL, TOOD 2021) with YOLO26's Small-Target-Aware extension (STAL).

align_metric = score^alpha * IoU^beta              (Feng et al., TOOD, ICCV 2021)
STAL: for the *candidate* test only, a GT side d < s_min is replaced by s_ref (centre kept),
      so tiny boxes still contain at least one stride-8 anchor centre.  Scoring, the final
      assignment and the regression target all use the original box.  (YOLO26 paper, Eq. 5:
      s_min = 8 = smallest stride, s_ref = 16 = next stride.)
"""

import torch
import torch.nn as nn

from .utils import bbox_iou


class TaskAlignedAssigner(nn.Module):
    def __init__(self, nc, topk=10, topk2=None, alpha=0.5, beta=6.0, stal=True, s_min=8.0, s_ref=16.0, eps=1e-9):
        super().__init__()
        self.nc, self.topk, self.topk2 = nc, topk, topk2
        self.alpha, self.beta = alpha, beta
        self.stal, self.s_min, self.s_ref = stal, s_min, s_ref
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Args (all in input-image pixels):
            pd_scores (B, A, nc) sigmoid scores        pd_bboxes (B, A, 4) xyxy
            anc_points (A, 2)                          gt_labels (B, M, 1)
            gt_bboxes (B, M, 4) xyxy                   mask_gt (B, M, 1) 1 = real GT, 0 = padding
        Returns:
            target_bboxes (B, A, 4), target_scores (B, A, nc), fg_mask (B, A) bool
        """
        bs, n_anchors = pd_scores.shape[:2]
        n_max = gt_bboxes.shape[1]
        device = pd_scores.device
        if n_max == 0:
            return (torch.zeros_like(pd_bboxes), torch.zeros_like(pd_scores),
                    torch.zeros(bs, n_anchors, dtype=torch.bool, device=device))

        # 1) candidates: anchor centre inside the (STAL-inflated) GT box
        cand_boxes = self._stal_boxes(gt_bboxes) if self.stal else gt_bboxes
        mask_in_gts = self._in_boxes(anc_points, cand_boxes)  # (B, M, A)

        # 2) alignment metric on candidates
        align_metric, overlaps = self._metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)

        # 3) top-k per GT, 4) resolve anchors claimed by several GTs (keep highest IoU)
        mask_topk = self._topk(align_metric, self.topk, mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        mask_pos = self._resolve(mask_pos, overlaps)

        # 5) optional second top-k after conflicts are resolved (YOLO26 one-to-one branch: topk=7 -> topk2=1)
        if self.topk2 is not None and self.topk2 < self.topk:
            m = align_metric * mask_pos
            keep = self._topk(m, self.topk2, mask_gt.expand(-1, -1, self.topk2).bool())
            mask_pos = mask_pos * keep

        fg_mask = mask_pos.sum(-2) > 0  # (B, A)
        target_gt_idx = mask_pos.argmax(-2)  # (B, A)

        # 6) targets
        batch_ind = torch.arange(bs, device=device)[:, None]
        flat_idx = target_gt_idx + batch_ind * n_max
        target_labels = gt_labels.long().flatten()[flat_idx].clamp_(0)  # (B, A)
        target_bboxes = gt_bboxes.view(-1, 4)[flat_idx]  # (B, A, 4)
        target_scores = torch.zeros(bs, n_anchors, self.nc, dtype=pd_scores.dtype, device=device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        target_scores = torch.where(fg_mask[..., None], target_scores, torch.zeros_like(target_scores))

        # soft target: normalised alignment (best anchor of each GT gets that GT's best IoU)
        align_metric = align_metric * mask_pos
        pos_align = align_metric.amax(-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(-1, keepdim=True)
        norm = (align_metric * pos_overlaps / (pos_align + self.eps)).amax(-2).unsqueeze(-1)
        return target_bboxes, target_scores * norm, fg_mask

    # ------------------------------------------------------------------ helpers
    def _stal_boxes(self, gt):
        """Paper Eq. 5: d' = s_ref if d < s_min else d, for width and height, centre unchanged."""
        cx, cy = (gt[..., 0] + gt[..., 2]) / 2, (gt[..., 1] + gt[..., 3]) / 2
        w, h = gt[..., 2] - gt[..., 0], gt[..., 3] - gt[..., 1]
        w = torch.where(w < self.s_min, torch.full_like(w, self.s_ref), w)
        h = torch.where(h < self.s_min, torch.full_like(h, self.s_ref), h)
        return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), -1)

    def _in_boxes(self, xy, boxes, eps=1e-9):
        bs, n_boxes, _ = boxes.shape
        lt, rb = boxes.view(-1, 1, 4).chunk(2, 2)
        deltas = torch.cat((xy[None] - lt, rb - xy[None]), 2).view(bs, n_boxes, xy.shape[0], -1)
        return deltas.amin(3).gt_(eps).to(boxes.dtype)

    def _metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask):
        bs, n_max = gt_bboxes.shape[:2]
        n_anchors = pd_bboxes.shape[1]
        mask = mask.bool()
        overlaps = torch.zeros(bs, n_max, n_anchors, dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros_like(overlaps)

        b_idx = torch.arange(bs, device=pd_scores.device).view(-1, 1).expand(-1, n_max)
        cls_idx = gt_labels.squeeze(-1).long().clamp(0)
        bbox_scores[mask] = pd_scores[b_idx, :, cls_idx][mask]  # score of the GT's class at each anchor

        pd = pd_bboxes.unsqueeze(1).expand(-1, n_max, -1, -1)[mask]
        gt = gt_bboxes.unsqueeze(2).expand(-1, -1, n_anchors, -1)[mask]
        overlaps[mask] = bbox_iou(gt, pd, CIoU=True).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    @staticmethod
    def _topk(metrics, k, topk_mask):
        k = min(k, metrics.shape[-1])
        topk_metrics, topk_idxs = torch.topk(metrics, k, dim=-1, largest=True)
        topk_idxs.masked_fill_(~topk_mask[..., :k], 0)
        count = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8)
        for i in range(k):
            count.scatter_add_(-1, topk_idxs[:, :, i : i + 1], ones)
        count.masked_fill_(count > 1, 0)  # index 0 repeated from masking -> invalid
        return count.to(metrics.dtype)

    @staticmethod
    def _resolve(mask_pos, overlaps):
        fg = mask_pos.sum(-2)
        if fg.max() > 1:
            multi = (fg.unsqueeze(1) > 1).expand(-1, mask_pos.shape[1], -1)
            best = overlaps.argmax(1)
            is_best = torch.zeros_like(mask_pos)
            is_best.scatter_(1, best.unsqueeze(1), 1)
            mask_pos = torch.where(multi, is_best, mask_pos)
        return mask_pos
