import mne
import numpy as np

def compute_relative_band_power(epoch_data, sfreq, bands):
    psd, freqs = mne.time_frequency.psd_array_welch(
        epoch_data, sfreq=sfreq, fmin=0.5, fmax=30, n_per_seg=256
    )
    total_power = np.sum(psd, axis=-1, keepdims=True)
    band_powers = {}
    for band_name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        band_power = np.sum(psd[:, idx], axis=-1)
        band_powers[band_name] = band_power / total_power.flatten()
    return band_powers
