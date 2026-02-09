from src.xml_loader import load_its
import argparse

def print_ctc(its_path):
    root = load_its(its_path)

    conversations = root.findall(".//Conversation")

    total_turns = 0
    for c in conversations:
        turns = c.get("turnTaking", "0")
        if turns.isdigit():
            total_turns += int(turns)

    print("=== CTC (Conversational Turns) ===")
    print("Total conversations:", len(conversations))
    print("Total turns:", total_turns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CTC from a LENA ITS file")
    parser.add_argument("its_path", help="Path to the .its file")
    args = parser.parse_args()

    print_ctc(args.its_path)
