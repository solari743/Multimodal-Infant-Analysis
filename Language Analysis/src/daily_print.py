import argparse
from src.daily_core import compute_daily_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print daily AWC, CVC, CTC")
    parser.add_argument("data_folder", help="Folder with lena CSVs")
    args = parser.parse_args()

    summary = compute_daily_summary(args.data_folder)

    print("\n=== DAILY SUMMARY ===")
    print(summary)
