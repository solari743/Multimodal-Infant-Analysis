import numpy as np
import pandas as pd
from src.h5_loader import load_accel

def compute_speed_epochs(h5_path, target_id, epoch="30s"):
    acc, time_dt, time_raw = load_accel(h5_path, target_id)

    time_sec = (time_raw - time_raw[0]) * 1e-6
    dt = np.gradient(time_sec)
    acc_mag = np.linalg.norm(acc, axis=1)
    speed_est = np.cumsum(acc_mag * dt)

    df = pd.DataFrame({"timestamp": time_dt, "speed_est": speed_est}).set_index("timestamp")
    feats = df["speed_est"].resample(epoch).agg(["mean", "std", "min", "max"]).dropna()
    return feats
