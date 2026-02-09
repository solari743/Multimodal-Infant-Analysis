import pandas as pd
from src.core.magnitude_core import compute_magnitude_epochs

def compute_cov_epochs(h5_path, target_id, epoch="30s"):
    feats = compute_magnitude_epochs(h5_path, target_id, epoch)
    cov = (feats["std"] / feats["mean"]).replace([float("inf"), -float("inf")], 0)
    return pd.DataFrame({"cov": cov}).dropna()
