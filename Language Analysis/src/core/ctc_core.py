import pandas as pd
import os

def compute_ctc(data_folder, output_csv):
    conversations_path = os.path.join(data_folder, "conversations.csv")
    if not os.path.exists(conversations_path):
        raise FileNotFoundError("conversations.csv not found. Run extractor first.")

    df = pd.read_csv(conversations_path)

    df['turn_count'] = pd.to_numeric(df['turn_count'], errors='coerce').fillna(0)

    total_ctc = df['turn_count'].sum()

    result = pd.DataFrame({
        "metric": ["Conversational Turn Count (CTC)"],
        "value": [total_ctc]
    })

    result.to_csv(output_csv, index=False)

    print("=== CTC RESULTS ===")
    print(result)

    return result
