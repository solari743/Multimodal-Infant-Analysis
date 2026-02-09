import argparse, os
import matplotlib.pyplot as plt
from src.core.magnitude_core import compute_magnitude_epochs

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot mean magnitude per epoch")
    p.add_argument("h5_path")
    p.add_argument("target_id")
    p.add_argument("--out", default="Graphs")
    args = p.parse_args()

    df = compute_magnitude_epochs(args.h5_path, args.target_id)
    os.makedirs(args.out, exist_ok=True)

    plt.figure(figsize=(12,5))
    plt.plot(df.index, df["mean"], label="Mean Magnitude")
    plt.title("Mean Acceleration Magnitude (per epoch)")
    plt.xlabel("Time"); plt.ylabel("Magnitude"); plt.legend(); plt.tight_layout()

    out_png = os.path.join(args.out, "magnitude_mean.png")
    plt.savefig(out_png, dpi=200); plt.close()
    print(f"Saved plot to {out_png}")
