"""Train YOLO26 from scratch on a KITTI subset.

    python train.py --cfg configs/kitti_yolo26n.yaml
    python train.py --cfg configs/kitti_yolo26n.yaml --name no_stal assign.stal=false
"""

import argparse
import csv
import os
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from val import evaluate, print_results
from yolo26.config import apply_overrides, dump_config, load_config
from yolo26.data import YOLODataset, build_loader
from yolo26.loss import YOLO26Loss
from yolo26.model import build_model, model_info
from yolo26.optim import build_optimizer
from yolo26.utils import ModelEMA, increment_path, seed_everything, select_device


def save_atomic(obj, path):
    """Write to a temp file, then rename: a run killed mid-save (e.g. Colab disconnect) never leaves a broken file."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/kitti_yolo26n.yaml")
    ap.add_argument("--device", default="", help="'' (auto), cpu, 0, mps")
    ap.add_argument("--name", default="exp", help="run folder: runs/train/<name>")
    ap.add_argument("--resume", default="", help="path to runs/train/<name>/weights/last.pt to continue a stopped run")
    ap.add_argument("overrides", nargs="*", help="section.key=value, e.g. train.epochs=50 assign.stal=false")
    args = ap.parse_args()

    device = select_device(args.device)
    resume_ckpt = None
    if args.resume:  # continue in the same folder with the same config
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        cfg = apply_overrides(resume_ckpt["cfg"], args.overrides)  # e.g. a new data.root after a Colab restart
        save_dir = Path(args.resume).resolve().parent.parent
        print(f"resuming {save_dir} from epoch {resume_ckpt['epoch']}")
    else:
        cfg = load_config(args.cfg, args.overrides)
        save_dir = increment_path(Path("runs/train") / args.name)
        (save_dir / "weights").mkdir(parents=True, exist_ok=True)
        dump_config(cfg, save_dir / "config.yaml")
    C_data, C_model, C_train = cfg["data"], cfg["model"], cfg["train"]
    seed_everything(C_train["seed"])
    names, nc = C_data["names"], len(C_data["names"])
    imgsz = tuple(C_train["imgsz"])
    assert imgsz[0] % 32 == 0 and imgsz[1] % 32 == 0, "imgsz must be multiples of 32"

    # ---------------- data
    train_ds = YOLODataset(C_data["root"], C_data["train"], imgsz, C_model["in_channels"], augment=True,
                           hyp=cfg["augment"])
    val_ds = YOLODataset(C_data["root"], C_data["val"], imgsz, C_model["in_channels"], augment=False)
    train_loader = build_loader(train_ds, C_train["batch"], C_train["workers"], shuffle=True)
    val_loader = build_loader(val_ds, C_train["batch"] * 2, C_train["workers"], shuffle=False)
    print(f"train {len(train_ds)} images, val {len(val_ds)} images, classes {names}")

    # ---------------- model
    model = build_model(cfg).to(device)
    model.head.bias_init(imgsz)
    n_p, gflops = model_info(model, imgsz, C_model["in_channels"])
    print(f"YOLO26{C_model['scale']}{' + BoTNet' if C_model.get('botnet') else ''}: {n_p / 1e6:.2f}M params (training, both heads)"
          + (f", {gflops:.1f} GFLOPs @ {imgsz[0]}x{imgsz[1]}" if gflops else ""))
    model.train()
    criterion = YOLO26Loss(model, cfg["loss"], cfg["assign"])
    ema = ModelEMA(model)

    # ---------------- optimiser + schedule
    epochs = C_train["epochs"]
    opt = build_optimizer(model, C_train["optimizer"], C_train["lr0"], C_train["momentum"],
                          C_train["weight_decay"], C_train["muon_w"], C_train["sgd_w"])
    lf = lambda e: (1 - e / epochs) * (1.0 - C_train["lrf"]) + C_train["lrf"]  # noqa: E731  linear decay
    nb = len(train_loader)
    n_warm = max(round(C_train["warmup_epochs"] * nb), 100) if C_train["warmup_epochs"] > 0 else 0
    use_amp = C_train["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    csv_path = save_dir / "results.csv"
    header = ["epoch", "lr", "o2m_weight", "box", "cls", "l1", "loss_o2m", "loss_o2o",
              "P", "R", "mAP50", "mAP50-95", "ms_per_img", "time_s"]
    best, best_epoch, start_epoch = -1.0, 0, 0
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["train_model"])
        ema.ema.load_state_dict(resume_ckpt["model"])
        ema.updates = resume_ckpt["ema_updates"]
        opt.load_state_dict(resume_ckpt["optimizer"])
        if use_amp and resume_ckpt.get("scaler"):
            scaler.load_state_dict(resume_ckpt["scaler"])
        start_epoch, best, best_epoch = resume_ckpt["epoch"], resume_ckpt["best"], resume_ckpt["best_epoch"]
        if start_epoch >= epochs:
            print("this run already finished all epochs")
            return
    else:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    branch = "one2one" if C_model["end2end"] else "one2many"
    t_start = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        criterion.set_epoch(epoch, epochs)
        if epoch >= epochs - C_train["close_mosaic"] and train_ds.mosaic:
            print("closing Mosaic for the last epochs")
            train_ds.mosaic = False
            train_loader = build_loader(train_ds, C_train["batch"], C_train["workers"], shuffle=True)  # workers copy the flag

        mloss = torch.zeros(5, device=device)
        pbar = tqdm(enumerate(train_loader), total=nb, desc=f"epoch {epoch + 1}/{epochs}")
        t_ep = time.time()
        for i, (imgs, targets, _) in pbar:
            ni = i + nb * epoch
            # warm-up: lr 0 -> lr0 (bias group 0.1 -> lr0), momentum 0.8 -> momentum
            if ni <= n_warm:
                for j, g in enumerate(opt.param_groups):
                    start = 0.1 if j == 2 else 0.0
                    g["lr"] = start + (g["initial_lr"] * lf(epoch) - start) * ni / max(n_warm, 1)
                    if "momentum" in g:
                        g["momentum"] = 0.8 + (C_train["momentum"] - 0.8) * ni / max(n_warm, 1)
            else:
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * lf(epoch)

            imgs = imgs.to(device, non_blocking=True).float() / 255
            with torch.autocast(device_type=device.type, enabled=use_amp):
                preds = model(imgs)
            loss, items = criterion({k: (tuple(t.float() for t in v) if isinstance(v, tuple) else v)
                                     for k, v in preds.items()}, targets)
            if not torch.isfinite(loss):
                print("non-finite loss, skipping batch")
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            ema.update(model)

            mloss = (mloss * i + items) / (i + 1)
            pbar.set_postfix(box=f"{mloss[0]:.3f}", cls=f"{mloss[1]:.3f}", l1=f"{mloss[2]:.3f}",
                             o2m_w=f"{criterion.alpha:.2f}", lr=f"{opt.param_groups[1]['lr']:.2e}")

        # ---------------- validation (EMA weights)
        last_epoch = epoch == epochs - 1
        if (epoch + 1) % C_train["val_interval"] == 0 or last_epoch:
            res = evaluate(ema.ema, val_loader, nc, device, branch=branch)
            print(f"  val {branch}: P {res['P']:.3f}  R {res['R']:.3f}  mAP50 {res['mAP50']:.3f}  "
                  f"mAP50-95 {res['mAP50-95']:.3f}")
        else:
            res = {"P": float("nan"), "R": float("nan"), "mAP50": float("nan"), "mAP50-95": float("nan"),
                   "ms_per_img": float("nan")}

        row = [epoch + 1, opt.param_groups[1]["lr"], round(criterion.alpha, 4), *[round(x, 5) for x in mloss.tolist()],
               *[round(float(res[k]), 5) for k in ("P", "R", "mAP50", "mAP50-95", "ms_per_img")],
               round(time.time() - t_ep, 1)]
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        fitness = res["mAP50-95"]
        improved = fitness == fitness and fitness > best  # not NaN
        if improved:
            best, best_epoch = fitness, epoch + 1
        ckpt = {"epoch": epoch + 1, "model": ema.ema.state_dict(), "cfg": cfg, "metrics": res}
        if improved:
            save_atomic(ckpt, save_dir / "weights" / "best.pt")  # small: EMA weights only
        # last.pt also keeps everything needed for --resume
        ckpt.update(train_model=model.state_dict(), optimizer=opt.state_dict(), ema_updates=ema.updates,
                    scaler=scaler.state_dict() if use_amp else None, best=best, best_epoch=best_epoch)
        save_atomic(ckpt, save_dir / "weights" / "last.pt")
        if C_train["patience"] and epoch + 1 - best_epoch >= C_train["patience"]:
            print(f"early stop: no improvement for {C_train['patience']} epochs")
            break

    print(f"\ndone in {(time.time() - t_start) / 3600:.2f} h. best mAP50-95 {best:.4f} at epoch {best_epoch}")
    if (save_dir / "weights" / "best.pt").exists():
        ck = torch.load(save_dir / "weights" / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print_results(evaluate(model, val_loader, nc, device, branch=branch), names)
        if C_model["end2end"]:
            print("\none-to-many head + NMS (for comparison):")
            print_results(evaluate(model, val_loader, nc, device, branch="one2many"), names)
    print(f"results saved to {save_dir}")


if __name__ == "__main__":
    main()
