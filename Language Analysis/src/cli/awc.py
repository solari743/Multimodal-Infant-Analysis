import argparse
from src.core.awc_core import compute_awc

def main():
    parser = argparse.ArgumentParser(description="Compute Adult Word Count (AWC)")
    parser.add_argument("--data-folder", default="lena_outputs", help="Folder with extracted CSVs")
    parser.add_argument("--out", default="Graphs/awc.csv", help="Output CSV path")

    args = parser.parse_args()

    compute_awc(args.data_folder, args.out)

if __name__ == "__main__":
    main()
