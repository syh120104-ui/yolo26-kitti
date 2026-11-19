"""Plot loss / mAP curves from one or more runs (handy for ablation figures).

    python tools/plot_results.py runs/train/exp runs/train/no_stal runs/train/sgd
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(run):
    with open(Path(run) / "results.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="runs/compare.png")
    args = ap.parse_args()

    keys = ["box", "cls", "l1", "mAP50", "mAP50-95", "o2m_weight"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for run in args.runs:
        d = read(run)
        for ax, k in zip(axes.flat, keys):
            ax.plot(d["epoch"], d[k], label=Path(run).name)
            ax.set_title(k)
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
