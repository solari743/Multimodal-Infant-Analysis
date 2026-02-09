import pandas as pd
from src.csv_loader import load_lena_csvs

def compute_daily_summary(data_folder):
    """
    Load LENA CSVs and compute daily totals for:
    - AWC (Adult Word Count)
    - CVC (Child Vocalizations)
    - CTC (Conversational Turns)
    Returns a pandas DataFrame with columns: date, AWC, CVC, CTC
    """
    recording_info, segments, conversations = load_lena_csvs(data_folder)

    recording_start_time = pd.to_datetime(recording_info["start_clock_time"].iloc[0])

    # ---- Segments: AWC + CVC ----
    seg = segments.copy()
    seg["time_offset"] = pd.to_timedelta(seg["start_time"], errors="coerce")
    seg.dropna(subset=["time_offset"], inplace=True)

    seg["absolute_time"] = seg["time_offset"] + recording_start_time
    seg["date"] = seg["absolute_time"].dt.date

    # Compute AWC per segment
    seg["AWC"] = seg["fem_adult_word_cnt"].fillna(0) + seg["male_adult_word_cnt"].fillna(0)

    daily_awc_cvc = seg.groupby("date").agg(
        AWC=("AWC", "sum"),
        CVC=("child_utt_count", "sum")
    ).reset_index()

    # ---- Conversations: CTC ----
    conv = conversations.copy()
    conv["time_offset"] = pd.to_timedelta(conv["start_time"], errors="coerce")
    conv.dropna(subset=["time_offset"], inplace=True)

    conv["absolute_time"] = conv["time_offset"] + recording_start_time
    conv["date"] = conv["absolute_time"].dt.date

    daily_ctc = conv.groupby("date").agg(
        CTC=("turn_count", "sum")
    ).reset_index()

    daily = pd.merge(daily_awc_cvc, daily_ctc, on="date", how="outer").fillna(0)
    return daily
