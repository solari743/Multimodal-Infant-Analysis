import pandas as pd
import os

def compute_awc(data_folder, output_csv):
    segments_path = os.path.join(data_folder, "segments.csv")
    if not os.path.exists(segments_path):
        raise FileNotFoundError("segments.csv not found. Run extractor first.")

    df = pd.read_csv(segments_path)

    df['fem_adult_word_cnt'] = pd.to_numeric(df['fem_adult_word_cnt'], errors='coerce').fillna(0)
    df['male_adult_word_cnt'] = pd.to_numeric(df['male_adult_word_cnt'], errors='coerce').fillna(0)

    df['AWC'] = df['fem_adult_word_cnt'] + df['male_adult_word_cnt']

    total_awc = df['AWC'].sum()

    result = pd.DataFrame({
        "metric": ["Adult Word Count (AWC)"],
        "value": [total_awc]
    })

    result.to_csv(output_csv, index=False)

    print("=== AWC RESULTS ===")
    print(result)

    return result
