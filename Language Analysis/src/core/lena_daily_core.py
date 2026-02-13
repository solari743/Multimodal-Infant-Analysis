import pandas as pd
import os

class LENADailySummarizer:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.recording_info = None
        self.segments = None
        self.conversations = None
        self.daily_summary = None

    def load_data(self):
        print(f"Loading data from {self.data_folder}...")
        try:
            self.recording_info = pd.read_csv(os.path.join(self.data_folder, "recording_info.csv"))
            self.segments = pd.read_csv(os.path.join(self.data_folder, "segments.csv"))
            self.conversations = pd.read_csv(os.path.join(self.data_folder, "conversations.csv"))
            print("Loaded recording info, segments, and conversations.")
            return True
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return False

    def calculate_daily_totals(self):
        start_time_column = 'start_clock_time'
        try:
            recording_start_time = pd.to_datetime(self.recording_info[start_time_column].iloc[0])
        except (KeyError, IndexError):
            print(f"Error: Column '{start_time_column}' not found in recording_info.csv.")
            return False

        daily_awc_cvc = pd.DataFrame()
        daily_ctc = pd.DataFrame()

        if self.segments is not None:
            segments_df = self.segments.copy()
            segments_df['time_offset'] = pd.to_timedelta(segments_df['start_time'], errors='coerce')
            segments_df.dropna(subset=['time_offset'], inplace=True)
            segments_df['absolute_time'] = segments_df['time_offset'].apply(lambda x: recording_start_time + x)
            segments_df['date'] = segments_df['absolute_time'].dt.date
            segments_df['AWC'] = segments_df['fem_adult_word_cnt'].fillna(0) + segments_df['male_adult_word_cnt'].fillna(0)
            daily_awc_cvc = segments_df.groupby('date').agg(AWC=('AWC', 'sum'), CVC=('child_utt_count', 'sum')).reset_index()

        if self.conversations is not None:
            conv_df = self.conversations.copy()
            conv_df['time_offset'] = pd.to_timedelta(conv_df['start_time'], errors='coerce')
            conv_df.dropna(subset=['time_offset'], inplace=True)
            conv_df['absolute_time'] = conv_df['time_offset'].apply(lambda x: recording_start_time + x)
            conv_df['date'] = conv_df['absolute_time'].dt.date
            daily_ctc = conv_df.groupby('date').agg(CTC=('turn_count', 'sum')).reset_index()

        if not daily_awc_cvc.empty:
            self.daily_summary = daily_awc_cvc
            if not daily_ctc.empty:
                self.daily_summary = pd.merge(self.daily_summary, daily_ctc, on='date', how='outer').fillna(0)
            else:
                self.daily_summary['CTC'] = 0
        elif not daily_ctc.empty:
            self.daily_summary = daily_ctc
            self.daily_summary['AWC'] = 0
            self.daily_summary['CVC'] = 0
        else:
            print("No data available to create a daily summary.")
            return False

        print("\nSuccessfully calculated daily totals:")
        print(self.daily_summary)
        return True
