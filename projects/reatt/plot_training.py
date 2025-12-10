import argparse
import csv
import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_loss_csv(loss_csv_path: str) -> Tuple[List[int], List[float]]:
    steps: List[int] = []
    losses: List[float] = []
    with open(loss_csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                step = int(row[0])
                loss = float(row[1])
            except ValueError:
                continue
            steps.append(step)
            losses.append(loss)
    return steps, losses


def plot_curve(steps: List[int], losses: List[float], out_path: str, title: str = "Training Curve (NLL)") -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, losses, label="train NLL", color="tab:blue")
    plt.xlabel("Step")
    plt.ylabel("Negative Log-Likelihood (cross-entropy)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training loss vs steps from loss.csv")
    parser.add_argument("--work_dir", type=str, required=True, help="Work dir containing loss.csv")
    parser.add_argument("--loss_csv", type=str, default="loss.csv", help="Filename of the loss CSV inside work_dir")
    parser.add_argument("--out", type=str, default="loss_curve.png", help="Output image filename (inside work_dir)")
    args = parser.parse_args()

    loss_csv_path = os.path.join(args.work_dir, args.loss_csv)
    out_path = os.path.join(args.work_dir, args.out)

    steps, losses = read_loss_csv(loss_csv_path)
    if not steps:
        raise SystemExit(f"No data found in {loss_csv_path}")
    plot_curve(steps, losses, out_path)
    print(f"Wrote plot: {out_path}")


if __name__ == "__main__":
    main()


