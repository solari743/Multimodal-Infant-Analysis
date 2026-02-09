import pandas as pd
import os

def load_lena_csvs(folder):
    recording_info = pd.read_csv(os.path.join(folder, "recording_info.csv"))
    segments = pd.read_csv(os.path.join(folder, "segments.csv"))
    conversations = pd.read_csv(os.path.join(folder, "conversations.csv"))
    return recording_info, segments, conversations
