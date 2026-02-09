from src.xml_loader import load_its
import argparse

def print_cvc(its_path):
    root = load_its(its_path)
    segments = root.findall(".//Segment")

    total_cvc = 0

    for s in segments:
        cnt = s.get("childUttCnt", "0")
        try:
            total_cvc += int(cnt)
        except:
            pass

    print("=== CVC (Child Vocalizations) ===")
    print("Total child utterances:", total_cvc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CVC from a LENA ITS file")
    parser.add_argument("its_path", help="Path to the .its file")
    args = parser.parse_args()

    print_cvc(args.its_path)