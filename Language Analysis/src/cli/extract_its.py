import argparse
from src.core.lena_extractor_core import LENAExtractor

def main():
    parser = argparse.ArgumentParser(description="Extract data from LENA ITS file")
    parser.add_argument("its_file", help="Path to .its file")
    parser.add_argument("--out", default="lena_outputs", help="Output directory for CSV files")

    args = parser.parse_args()

    extractor = LENAExtractor(args.its_file)

    if not extractor.parse_file():
        return

    extractor.extract_recording_info()
    extractor.extract_segments()
    extractor.extract_conversations()
    extractor.calculate_summary_stats()
    extractor.print_summary()
    extractor.save_all_data(args.out)

    print("\nExtraction complete.")

if __name__ == "__main__":
    main()
