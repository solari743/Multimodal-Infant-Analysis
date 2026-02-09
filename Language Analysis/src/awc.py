from src.xml_loader import load_its
import argparse

def print_awc(its_path):
    root = load_its(its_path)
    segments = root.findall(".//Segment")

    fem = 0
    male = 0

    for s in segments:
        try:
            fem += int(s.get("femAdultWordCnt", "0"))
            male += int(s.get("maleAdultWordCnt", "0"))
        except:
            pass

    print("=== AWC (Adult Word Count) ===")
    print("Female adult words:", fem)
    print("Male adult words:", male)
    print("Total adult words:", fem + male)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute AWC from a LENA ITS file")
    parser.add_argument("its_path", help="Path to the .its file")
    args = parser.parse_args()

    print_awc(args.its_path)
