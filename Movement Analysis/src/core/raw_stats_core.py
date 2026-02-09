import numpy as np
import pandas as pd
from scipy.stats import skew
from src.h5_loader import load_accel

def compute_raw_stats(h5_path, target_id):
    acc, time_dt, time_raw = load_accel(h5_path, target_id)

    mag = np.linalg.norm(acc, axis=1)

    time_sec = (time_raw - time_raw[0]) * 1e-6
    duration_sec = time_sec[-1] if len(time_sec) > 0 else 0

    stats = {
        "n_samples": len(mag),
        "duration_seconds": duration_sec,
        "duration_hours": duration_sec / 3600 if duration_sec > 0 else 0,
        "mean_magnitude": float(np.mean(mag)) if len(mag) else np.nan,
        "std_magnitude": float(np.std(mag)) if len(mag) else np.nan,
        "min_magnitude": float(np.min(mag)) if len(mag) else np.nan,
        "max_magnitude": float(np.max(mag)) if len(mag) else np.nan,
        "skewness": float(skew(mag)) if len(mag) > 2 else np.nan,
    }

    if stats["mean_magnitude"] and stats["mean_magnitude"] != 0:
        stats["cov"] = stats["std_magnitude"] / stats["mean_magnitude"]
    else:
        stats["cov"] = np.nan

    return pd.DataFrame([stats])
