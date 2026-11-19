"""Building blocks used by YOLO11 / YOLO26.

Block definitions follow the descriptions in the YOLO11 docs and the YOLO26 paper (Fig. S1)
and mirror the public Ultralytics implementation (https://github.com/ultralytics/ultralytics,
ultralytics/nn/modules/block.py). Re-implemented here so every line can be read and modified.
"""

import math

import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv2d -> BatchNorm -> SiLU."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depth-wise convolution (used in the light classification branch of the head, YOLOv10/11/26)."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    """CSP bottleneck with 3 convolutions (YOLOv5)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3 with a selectable kernel size (YOLO11)."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C2f(nn.Module):
    """CSP bottleneck with 2 convs where every inner block output is concatenated (YOLOv8)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    """Multi-head self-attention on a feature map, with a depth-wise conv positional term (YOLOv10 PSA)."""

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        self.qkv = Conv(dim, dim + nh_kd * 2, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q, k, v = self.qkv(x).view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class PSABlock(nn.Module):
    """Attention + FFN, both residual."""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C2PSA(nn.Module):
    """CSP block whose second half goes through PSA blocks (YOLO11/YOLO26, after SPPF at P5)."""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=max(1, self.c // 64)) for _ in range(n))
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


class C3k2(C2f):
    """C2f whose inner blocks are C3k (c3k=True) or Bottleneck; attn=True appends a PSABlock (YOLO26 last neck block)."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        blocks = []
        for _ in range(n):
            if attn:
                blocks.append(
                    nn.Sequential(
                        Bottleneck(self.c, self.c, shortcut, g),
                        PSABlock(self.c, attn_ratio=0.5, num_heads=max(1, self.c // 64)),
                    )
                )
            elif c3k:
                blocks.append(C3k(self.c, self.c, 2, shortcut, g))
            else:
                blocks.append(Bottleneck(self.c, self.c, shortcut, g))
        self.m = nn.ModuleList(blocks)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast: n chained max-pools == SPP(5, 9, 13) for n=3, k=5 (YOLOv5)."""

    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        for _ in range(self.n):
            y.append(self.m(y[-1]))
        out = self.cv2(torch.cat(y, 1))
        return out + x if self.add else out


# ----------------------------------------------------------------------------- BoTNet
class RelPosMHSA(nn.Module):
    """Multi-head self-attention with 2D relative position logits (BoTNet, Srinivas et al., CVPR 2021, Fig. 4).

    logits = q k^T + q r^T,  r(i,j -> k,l) = R_h[k - i] + R_w[l - j]   (content-content + content-position)
    The relative tables are learned for a reference feature size and linearly resized if the input differs.
    """

    def __init__(self, dim, heads=4, ref_hw=(8, 26)):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads, self.d = heads, dim // heads
        self.scale = self.d**-0.5
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        h, w = ref_hw
        self.rel_h = nn.Parameter(torch.randn(2 * h - 1, self.d) * self.d**-0.5)
        self.rel_w = nn.Parameter(torch.randn(2 * w - 1, self.d) * self.d**-0.5)

    @staticmethod
    def _table(t, n):
        if t.shape[0] == 2 * n - 1:
            return t
        return torch.nn.functional.interpolate(t.T[None], size=2 * n - 1, mode="linear", align_corners=True)[0].T

    @staticmethod
    def _rel_logits(q, table, n):
        """q (B, N, A, n, d) with the attended axis 'n' second-to-last; returns (B, N, A, n, n)."""
        logits = torch.einsum("...d,md->...m", q, table)  # (..., n, 2n-1)
        idx = torch.arange(n, device=q.device)
        rel = (idx[None, :] - idx[:, None] + n - 1)  # rel[i, k] = k - i + n - 1
        return logits.gather(-1, rel.expand(*logits.shape[:-1], n))

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).view(B, 3, self.heads, self.d, H * W).unbind(1)  # (B, N, d, HW)
        q = q * self.scale
        content = q.transpose(-2, -1) @ k  # (B, N, HW, HW)

        qs = q.view(B, self.heads, self.d, H, W).permute(0, 1, 3, 4, 2)  # (B, N, H, W, d)
        lw = self._rel_logits(qs, self._table(self.rel_w, W), W)  # (B, N, H, W, W)
        lh = self._rel_logits(qs.transpose(2, 3), self._table(self.rel_h, H), H).transpose(2, 3)  # (B, N, H, W, H)
        pos = (lh[..., :, None] + lw[..., None, :]).reshape(B, self.heads, H * W, H * W)

        attn = (content + pos).softmax(-1)
        out = v @ attn.transpose(-2, -1)  # (B, N, d, HW)
        return out.reshape(B, C, H, W)


class BoTBottleneck(nn.Module):
    """BoTNet block: the 3x3 conv of a ResNet bottleneck replaced by relative-position MHSA."""

    def __init__(self, c1, c2, heads=4, ref_hw=(8, 26), e=1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1)
        self.mhsa = RelPosMHSA(c_, heads, ref_hw)
        self.bn = nn.BatchNorm2d(c_, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)
        self.cv2 = Conv(c_, c2, 1, act=False)
        self.add = c1 == c2
        self.out_act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.cv2(self.act(self.bn(self.mhsa(self.cv1(x)))))
        return self.out_act(x + y) if self.add else self.out_act(y)


class C3k2BoT(C2f):
    """C3k2 (CSP) wrapper whose inner blocks are BoT bottlenecks — drop-in for the last backbone stage."""

    def __init__(self, c1, c2, n=1, e=0.5, heads=4, ref_hw=(8, 26)):
        super().__init__(c1, c2, n, False, 1, e)
        self.m = nn.ModuleList(BoTBottleneck(self.c, self.c, heads, ref_hw) for _ in range(n))
