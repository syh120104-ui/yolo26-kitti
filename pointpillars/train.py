"""Train PointPillars on KITTI LiDAR.

    python -m pointpillars.train --cfg pointpillars/kitti_pointpillars.yaml --name pp
    python -m pointpillars.train --resume runs/pointpillars/pp/weights/last.pt
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from yolo26.config import apply_overrides, dump_config, load_config
from yolo26.utils import increment_path, seed_everything, select_device

from .evaluate import evaluate, format_table
from .kitti import KittiLidar, build_loader
from .model import PillarLoss, PointPillars


def save_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def to_device(batch, device):
    batch["feats"] = batch["feats"].to(device, non_blocking=True)
    batch["coords"] = batch["coords"].to(device, non_blocking=True)
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="pointpillars/kitti_pointpillars.yaml")
    ap.add_argument("--name", default="pp")
    ap.add_argument("--device", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    device = select_device(args.device)
    ck = None
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        cfg = apply_overrides(ck["cfg"], args.overrides)
        save_dir = Path(args.resume).resolve().parent.parent
    else:
        cfg = load_config(args.cfg, args.overrides)
        save_dir = increment_path(Path("runs/pointpillars") / args.name)
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        dump_config(cfg, save_dir / "config.yaml")
    T = cfg["train"]
    seed_everything(T["seed"])

    train_ds = KittiLidar(cfg["data_root"], "train", cfg, augment=True)
    val_ds = KittiLidar(cfg["data_root"], "val", cfg, augment=False)
    loader = build_loader(train_ds, T["batch"], T["workers"], shuffle=True)
    print(f"train {len(train_ds)} frames, val {len(val_ds)} frames, classes {cfg['classes']}, "
          f"BEV grid {PointPillars(cfg).grid_hw}")

    model = PointPillars(cfg).to(device)
    print(f"PointPillars: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")
    criterion = PillarLoss(model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"], betas=(0.95, 0.99))
    epochs, nb = T["epochs"], len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=T["lr"], total_steps=epochs * nb, pct_start=0.4,
                                                div_factor=10, base_momentum=0.85, max_momentum=0.95)
    use_amp = T["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    csv_path = save_dir / "results.csv"
    head = ["epoch", "lr", "loc", "cls", "dir", "n_pos"] + [f"{c}_3d_mod" for c in cfg["classes"]] + \
        ["mean_3d_mod", "time_s"]
    start, best = 0, -1.0
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start, best = ck["epoch"], ck["best"]
        print(f"resuming {save_dir} from epoch {start}")
    else:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(head)

    for epoch in range(start, epochs):
        model.train()
        t0 = time.time()
        m = torch.zeros(3)
        npos = 0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        for i, batch in enumerate(pbar):
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(batch)
            loss, items, n = criterion(tuple(o.float() for o in out), batch)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            m = (m * i + items.cpu()) / (i + 1)
            npos += n
            pbar.set_postfix(loc=f"{m[0]:.3f}", cls=f"{m[1]:.3f}", dir=f"{m[2]:.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")

        res = None
        if (epoch + 1) % T["val_interval"] == 0 or epoch + 1 == epochs:
            res = evaluate(model, val_ds, device, batch=T["batch"], workers=T["workers"])
            table, mean_mod = format_table(res, cfg["classes"])
            print(table)
        mods = [res[(c, "Moderate", "3d")] if res else float("nan") for c in cfg["classes"]]
        mean_mod = float(np.nanmean(mods)) if res else float("nan")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, sched.get_last_lr()[0], *[round(float(x), 5) for x in m], npos,
                                    *[round(float(x), 4) for x in mods], round(mean_mod, 4), round(time.time() - t0, 1)])
        state = {"epoch": epoch + 1, "model": model.state_dict(), "cfg": cfg}
        if res is not None and mean_mod == mean_mod and mean_mod > best:
            best = mean_mod
            save_atomic(state, save_dir / "weights" / "best.pt")
        state.update(optimizer=opt.state_dict(), scheduler=sched.state_dict(), best=best)
        save_atomic(state, save_dir / "weights" / "last.pt")

    print(f"done. best mean 3D AP (Moderate) {100 * best:.2f}. results in {save_dir}")


if __name__ == "__main__":
    main()
