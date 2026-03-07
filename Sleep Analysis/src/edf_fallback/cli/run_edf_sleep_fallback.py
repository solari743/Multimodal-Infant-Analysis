import argparse

from core.edf_loader import (
    load_edf_file,
    extract_start_time,
    prepare_eeg_channels
)
from core.better_sleep_stages import (
    apply_ica,
    classify_sleep_stages,
    summarize_sleep_stages
)
from plot.plot_better_sleep_stages import plot_sleep_stages


def main():
    parser = argparse.ArgumentParser(
        description="Run fallback infant EEG sleep stage analysis from EDF"
    )

    parser.add_argument(
        "--edf",
        required=True,
        help="Path to the EDF file"
    )

    args = parser.parse_args()
    edf_file_path = args.edf

    raw = load_edf_file(edf_file_path)

    start_time = extract_start_time(raw)
    if start_time:
        print(f"Recording start time: {start_time}")
    else:
        print("Recording start time not available")

    raw, selected_channels = prepare_eeg_channels(raw)
    raw = apply_ica(raw, selected_channels)

    sleep_stages_smoothed, epoch_times = classify_sleep_stages(
        raw,
        start_time=start_time
    )

    summarize_sleep_stages(raw, sleep_stages_smoothed)
    plot_sleep_stages(epoch_times, sleep_stages_smoothed)


if __name__ == "__main__":
    main()