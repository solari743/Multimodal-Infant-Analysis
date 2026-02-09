import argparse, os
from src.core.magnitude_core import compute_magnitude_epochs

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute acceleration magnitude stats per epoch")
    p.add_argument("h5_path")
    p.add_argument("target_id")
    p.add_argument("--out", default="movement_outputs")
    args = p.parse_args()

    df = compute_magnitude_epochs(args.h5_path, args.target_id)
    print(df.head())

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "magnitude_epochs.csv")
    df.to_csv(out_csv)
    print(f"Saved to {out_csv}")
