import argparse
from src.core.cvc_core import compute_cvc

def main():
    parser = argparse.ArgumentParser(description="Compute Child Vocalization Count (CVC)")
    parser.add_argument("--data-folder", default="lena_outputs", help="Folder with extracted CSVs")
    parser.add_argument("--out", default="Graphs/cvc.csv", help="Output CSV path")

    args = parser.parse_args()

    compute_cvc(args.data_folder, args.out)

if __name__ == "__main__":
    main()
