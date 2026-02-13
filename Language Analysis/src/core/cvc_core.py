import pandas as pd
import os

def compute_cvc(data_folder, output_csv):
    segments_path = os.path.join(data_folder, "segments.csv")
    if not os.path.exists(segments_path):
        raise FileNotFoundError("segments.csv not found. Run extractor first.")

    df = pd.read_csv(segments_path)

    df['child_utt_count'] = pd.to_numeric(df['child_utt_count'], errors='coerce').fillna(0)

    total_cvc = df['child_utt_count'].sum()

    result = pd.DataFrame({
        "metric": ["Child Vocalization Count (CVC)"],
        "value": [total_cvc]
    })

    result.to_csv(output_csv, index=False)

    print("=== CVC RESULTS ===")
    print(result)

    return result
