import argparse
import os
import matplotlib.pyplot as plt
from src.daily_core import compute_daily_summary

def plot_daily(data_folder, output_dir="lena_visualizations"):
    summary = compute_daily_summary(data_folder)

    if summary.empty:
        print("No data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 8))
    summary.set_index("date").plot(
        kind="bar", y=["AWC", "CVC", "CTC"], ax=ax
    )

    ax.set_title("Total AWC, CVC, and CTC per Day")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Count")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "daily_summary_AWC_CVC_CTC.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot daily AWC, CVC, CTC from LENA CSVs")
    parser.add_argument("data_folder", help="Folder containing lena_extraction_output CSV files")
    parser.add_argument("--out", default="lena_visualizations", help="Output folder for plots")
    args = parser.parse_args()

    plot_daily(args.data_folder, args.out)
