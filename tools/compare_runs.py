"""Compare finished runs side by side (e.g. baseline vs BoTNet) and write a markdown table for the report.

    python tools/compare_runs.py runs/train/yolo26n runs/train/yolo26n_botnet
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from val import evaluate, load_checkpoint  # noqa: E402
from yolo26.data import YOLODataset, build_loader  # noqa: E402
from yolo26.model import model_info  # noqa: E402
from yolo26.utils import select_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run folders, e.g. runs/train/yolo26n")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="")
    ap.add_argument("--out", default="runs/compare.md")
    args = ap.parse_args()
    device = select_device(args.device)

    rows, names = [], None
    for run in args.runs:
        model, cfg = load_checkpoint(Path(run) / "weights" / "best.pt", device)
        names = cfg["data"]["names"]
        imgsz, in_ch = tuple(cfg["train"]["imgsz"]), cfg["model"]["in_channels"]
        ds = YOLODataset(args.data_root or cfg["data"]["root"], cfg["data"]["val"], imgsz, in_ch, augment=False)
        loader = build_loader(ds, 16, 0, shuffle=False)
        n_p, gflops = model_info(model, imgsz, in_ch)
        branches = ["one2one", "one2many"] if cfg["model"]["end2end"] else ["one2many"]
        for br in branches:
            res = evaluate(model, loader, len(names), device, branch=br)
            rows.append((Path(run).name, "NMS-free" if br == "one2one" else "NMS", n_p, gflops, res))

    head = ["run", "inference", "params (M)", "GFLOPs", "mAP50", "mAP50-95"] + [f"{n} AP50-95" for n in names] + ["ms/img"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for run, br, n_p, gflops, r in rows:
        per = [f"{r['per_class'][c][4]:.3f}" for c in range(len(names))]
        lines.append("| " + " | ".join([run, br, f"{n_p / 1e6:.2f}", f"{gflops:.2f}" if gflops else "-",
                                        f"{r['mAP50']:.3f}", f"{r['mAP50-95']:.3f}", *per, f"{r['ms_per_img']:.1f}"]) + " |")
    text = "\n".join(lines)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n\n(params count both heads during training; the deployed model keeps one)\n",
                              encoding="utf-8")
    print(text)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
