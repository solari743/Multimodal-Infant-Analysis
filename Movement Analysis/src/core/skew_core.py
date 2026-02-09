import numpy as np
import pandas as pd
from scipy.stats import skew
from src.h5_loader import load_accel

def compute_skew_epochs(h5_path, target_id, epoch="30s"):
    acc, time_dt, _ = load_accel(h5_path, target_id)
    mag = np.linalg.norm(acc, axis=1)
    df = pd.DataFrame({"timestamp": time_dt, "acc_magnitude": mag}).set_index("timestamp")

    def _sk(x):
        return skew(x) if len(x) >= 3 else np.nan

    out = df["acc_magnitude"].resample(epoch).apply(_sk).dropna()
    return out.to_frame(name="skewness")
