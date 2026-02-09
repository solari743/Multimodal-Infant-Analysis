import argparse
import os
from src.core.raw_stats_core import compute_raw_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute global raw movement statistics from an H5 file")
    parser.add_argument("h5_path", help="Path to the .h5 file")
    parser.add_argument("target_id", help="Sensor ID (e.g., XI-016162)")
    parser.add_argument("--out", default="movement_outputs", help="Output folder for CSV")
    args = parser.parse_args()

    df = compute_raw_stats(args.h5_path, args.target_id)

    print("\n=== RAW MOVEMENT SUMMARY STATS ===")
    print(df.T)  

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "raw_stats.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved raw stats to: {out_csv}")
