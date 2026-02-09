from src.xml_loader import load_its
import argparse

def print_other_stats(its_path):
    root = load_its(its_path)
    segments = root.findall(".//Segment")

    overlaps = 0

    for s in segments:
        try:
            overlaps += int(s.get("overlapCnt", "0"))
        except:
            pass

    print("=== OTHER STATS ===")
    print("Total overlaps:", overlaps)
    print("Total segments:", len(segments))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print other stats from a LENA ITS file")
    parser.add_argument("its_path", help="Path to the .its file")
    args = parser.parse_args()

    print_other_stats(args.its_path)