import mne
import numpy as np
import datetime
from scipy.ndimage import uniform_filter1d

frequency_bands = {
    "Delta" : (0.5, 3.0),
    "Theta" : (4, 8),
    "Alpha" : (8, 12),
}

def apply_ica(raw, selected_channels):
    ica = mne.preprocessing.ICA(n_components=min(20, len(selected_channels)), random_state=42,max_iter=300)

    try:
        ica.fit(raw)
        eog_indices, _ = ica.find_bads_eog(raw)
        ecg_indices, _ = ica.find_bads_ecg(raw)
        exclude_indices = list(set(eog_indices + ecg_indices))
        ica.exclude = exclude_indices
        ica.apply(raw)
    except Exception as e:
        print(f"ICA skipped due to error: {e}")

    return raw


def compute_relative_band_power(epoch_data, sfreq, bands):
    psd, freqs = mne.time_frequency.psd_array_welch(
        epoch_data,
        sfreq=sfreq,
        fmin=0.5,
        fmax=30,
        n_per_seg=256,
        verbose=False
    )

    total_power = np.sum(psd, axis=-1, keepdims=True)
    band_powers = {}

    for band_name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.sum(psd[:, idx], axis=-1)
        band_powers[band_name] = band_power / total_power.flatten()

    return band_powers


def classify_sleep_stages(raw, start_time=None, epoch_length_sec=30):
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    num_epochs = int(raw.times[-1] // epoch_length_sec)

    sleep_stages = []
    epoch_times = []

    if start_time:
        epoch_times = [
            start_time + datetime.timedelta(seconds=i * epoch_length_sec)
            for i in range(num_epochs)
        ]

    for i in range(num_epochs):
        start = int(i * epoch_length_sec * sfreq)
        end = int(min((i + 1) * epoch_length_sec * sfreq, data.shape[1]))

        epoch_data = data[:, start:end]

        band_powers = compute_relative_band_power(
            epoch_data,
            sfreq,
            frequency_bands
        )

        delta_mean = np.mean(band_powers["Delta"])
        theta_mean = np.mean(band_powers["Theta"])
        alpha_mean = np.mean(band_powers["Alpha"])

        if alpha_mean > 0.2 and delta_mean < 0.2:
            stage = 1
        elif 0.1 <= alpha_mean <= 0.2 and 0.15 <= theta_mean <= 0.3:
            stage = 3
        elif delta_mean > 0.3 and delta_mean > 1.5 * theta_mean:
            stage = 4
        else:
            stage = 2

        sleep_stages.append(stage)

    sleep_stages_smoothed = uniform_filter1d(
        sleep_stages,
        size=3,
        mode="nearest"
    )
    sleep_stages_smoothed = np.round(sleep_stages_smoothed).astype(int)

    return sleep_stages_smoothed, epoch_times


def summarize_sleep_stages(raw, sleep_stages_smoothed):
    stage_labels = {
        1: "Wake",
        2: "Movement",
        3: "Transitional",
        4: "NREM",
    }

    unique, counts = np.unique(sleep_stages_smoothed, return_counts=True)
    total_epochs = len(sleep_stages_smoothed)

    stage_percentages = {
        stage_labels.get(u, "Unknown"): (c / total_epochs) * 100
        for u, c in zip(unique, counts)
    }

    print("--- Infant EEG Sleep Stage Summary ---")
    print(f"Recording Duration: {raw.times[-1] / 3600:.2f} hours")
    print("Sleep Stage Distribution (%):")

    for stage_name, pct in stage_percentages.items():
        print(f" - {stage_name:12}: {pct:.2f}%")

    return stage_percentages, stage_labels