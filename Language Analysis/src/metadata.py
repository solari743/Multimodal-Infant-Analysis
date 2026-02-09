from src.xml_loader import load_its
import argparse

def print_metadata(its_path):
    root = load_its(its_path)

    processing = root.find(".//ProcessingUnit")
    recording = root.find(".//Recording")
    child = root.find(".//ChildInfo")

    print("===METADATA===")

    if processing is not None:
        start = recording.get("startTime")
        end = recording.get("endTime")
        print("Start Time: ", start)
        print("End Time: ", end)

        try:
            dur_sec = float(end) - float(start) / 1000
            print(f"Duration: {dur_sec/3600:.2f} hours")
        except:
            print("Duration: Unknown")

    if child is not None:
        print("Child age (months):", child.get("ageInMonths"))
        print("Child gender:", child.get("gender"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print metadata from a LENA ITS file")
    parser.add_argument("its_path", help="Path to the .its file")
    args = parser.parse_args()

    print_metadata(args.its_path)