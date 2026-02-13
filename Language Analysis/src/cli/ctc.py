import argparse
from src.core.ctc_core import compute_ctc

def main():
    parser = argparse.ArgumentParser(description="Compute Conversational Turn Count (CTC)")
    parser.add_argument("--data-folder", default="lena_outputs", help="Folder with extracted CSVs")
    parser.add_argument("--out", default="Graphs/ctc.csv", help="Output CSV path")

    args = parser.parse_args()

    compute_ctc(args.data_folder, args.out)

if __name__ == "__main__":
    main()
