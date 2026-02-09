import argparse, os
from src.core.skew_core import compute_skew_epochs

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute skewness per epoch")
    p.add_argument("h5_path")
    p.add_argument("target_id")
    p.add_argument("--out", default="movement_outputs")
    args = p.parse_args()

    df = compute_skew_epochs(args.h5_path, args.target_id)
    print(df.head())

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "skew_epochs.csv")
    df.to_csv(out_csv)
    print(f"Saved to {out_csv}")
