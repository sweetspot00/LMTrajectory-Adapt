import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot supervised inference result from saved npz.")
    parser.add_argument("--npz", type=str, required=True, help="Path to *_eval_outputs.npz saved by eval_accelerator.")
    parser.add_argument("--ped_idx", type=int, default=0, help="Pedestrian index to plot.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Prediction sample index to plot (0-based).")
    parser.add_argument("--output", type=str, default=None, help="Output png path. Defaults to <npz_dir>/ped<idx>_sample<idx>.png")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    data = np.load(npz_path, allow_pickle=True)
    obs = data["obs_traj"]
    gt = data["gt_traj"]
    preds = data["preds"]

    if args.ped_idx >= obs.shape[0]:
        raise IndexError(f"ped_idx {args.ped_idx} out of range (num ped {obs.shape[0]})")
    if args.sample_idx >= preds.shape[1]:
        raise IndexError(f"sample_idx {args.sample_idx} out of range (num samples {preds.shape[1]})")

    out_path = Path(args.output) if args.output else npz_path.parent / f"ped{args.ped_idx}_sample{args.sample_idx}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 5))
    plt.plot(obs[args.ped_idx, :, 0], obs[args.ped_idx, :, 1], "bo-", label="observed")
    plt.plot(gt[args.ped_idx, :, 0], gt[args.ped_idx, :, 1], "g^-", label="ground truth")
    plt.plot(preds[args.ped_idx, args.sample_idx, :, 0], preds[args.ped_idx, args.sample_idx, :, 1], color="orange", marker="o", label="predicted")
    plt.legend()
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
