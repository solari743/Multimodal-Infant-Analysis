import numpy as np
import pandas as pd
from src.h5_loader import load_accel

def compute_magnitude_epochs(h5_path, target_id, epoch="30s"):
    acc, time_dt, _ = load_accel(h5_path, target_id)
    mag = np.linalg.norm(acc, axis=1)

    df = pd.DataFrame({"timestamp": time_dt, "acc_magnitude": mag}).set_index("timestamp")
    feats = df["acc_magnitude"].resample(epoch).agg(["mean", "std", "min", "max"]).dropna()
    return feats
