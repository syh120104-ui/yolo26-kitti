"""MuSGD: hybrid Muon + SGD-momentum optimizer used to train YOLO26.

Paper (Sec. 3.3.1): weighted mixture of the Muon update and the SGD update for multi-dimensional
parameters (conv kernels, linear weights); pure SGD for 1-D parameters (biases, norm scales).
Details below follow the public Ultralytics implementation as documented at
https://docs.ultralytics.com/reference/optim/muon (source: ultralytics/optim/muon.py):
  * Muon momentum:  m <- beta*m + (1-beta)*g ; Nesterov: u = beta*m + (1-beta)*g
  * orthogonalise u with 5 quintic Newton-Schulz steps (a, b, c) = (3.4445, -4.7750, 2.0315)
    (coefficients from Keller Jordan's Muon), scale by sqrt(max(1, rows/cols))
  * step: p -= lr*muon_w * muon_update  and  p -= lr*sgd_w * sgd_update
  * weight decay only on the SGD part
"""

import torch
from torch import optim


def zeropower_via_newtonschulz5(G, steps: int = 5, eps: float = 1e-7):
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.is_cuda else G.float()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


def muon_update(grad, momentum, beta=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim > 2:  # conv kernel (out, in, kh, kw) -> (out, in*kh*kw)
        update = update.view(update.size(0), -1)
    update = zeropower_via_newtonschulz5(update)
    update = update * max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class MuSGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0, nesterov=True, muon=0.5, sgd=0.5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, use_muon=False)
        super().__init__(params, defaults)
        self.muon_w, self.sgd_w = muon, sgd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom, wd, nest = group["lr"], group["momentum"], group["weight_decay"], group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "sgd_buf" not in state:
                    state["sgd_buf"] = torch.zeros_like(p)
                # SGD part (computed before p changes)
                g_sgd = g.add(p, alpha=wd) if wd else g
                buf = state["sgd_buf"].mul_(mom).add_(g_sgd)
                sgd_step = g_sgd.add(buf, alpha=mom) if nest else buf

                if group["use_muon"] and p.ndim >= 2:
                    if "muon_buf" not in state:
                        state["muon_buf"] = torch.zeros_like(p)
                    upd = muon_update(g, state["muon_buf"], beta=mom, nesterov=nest)
                    p.add_(upd.view_as(p).to(p.dtype), alpha=-lr * self.muon_w)
                    p.add_(sgd_step, alpha=-lr * self.sgd_w)
                else:
                    p.add_(sgd_step, alpha=-lr)
        return loss


def build_optimizer(model, name="MuSGD", lr=0.01, momentum=0.94, weight_decay=5e-4, muon_w=0.5, sgd_w=0.5):
    """Three groups, as in YOLOv5-YOLO26: conv/linear weights (decay, Muon), norm weights, biases (no decay)."""
    g_w, g_bn, g_b = [], [], []
    for module in model.modules():
        for pname, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if pname == "bias":
                g_b.append(p)
            elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.GroupNorm)):
                g_bn.append(p)
            else:
                g_w.append(p)

    name = name.lower()
    if name == "musgd":
        opt = MuSGD([{"params": g_bn, "weight_decay": 0.0}], lr=lr, momentum=momentum, nesterov=True,
                    muon=muon_w, sgd=sgd_w)
        opt.add_param_group({"params": g_w, "weight_decay": weight_decay, "use_muon": True})
        opt.add_param_group({"params": g_b, "weight_decay": 0.0})
    elif name == "sgd":
        opt = optim.SGD(g_bn, lr=lr, momentum=momentum, nesterov=True)
        opt.add_param_group({"params": g_w, "weight_decay": weight_decay})
        opt.add_param_group({"params": g_b, "weight_decay": 0.0})
    elif name == "adamw":
        opt = optim.AdamW(g_bn, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        opt.add_param_group({"params": g_w, "weight_decay": weight_decay})
        opt.add_param_group({"params": g_b, "weight_decay": 0.0})
    else:
        raise ValueError(f"unknown optimizer {name}")
    for g in opt.param_groups:
        g["initial_lr"] = lr
    return opt
