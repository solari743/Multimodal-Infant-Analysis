import h5py
import numpy as np
import datetime

def load_accel(h5_path, target_id):
    with h5py.File(h5_path, "r") as f:
        if target_id not in f["Sensors"]:
            raise KeyError(f"Sensor ID {target_id} not found. Available: {list(f['Sensors'].keys())}")

        base = f"Sensors/{target_id}"
        if "Accelerometer" not in f[base] or "Time" not in f[base]:
            raise KeyError("Missing Accelerometer or Time dataset")

        acc = np.array(f[f"{base}/Accelerometer"][:], dtype=np.float64)
        time_raw = np.array(f[f"{base}/Time"][:], dtype=np.float64)

    time_dt = [datetime.datetime.fromtimestamp(t * 1e-6) for t in time_raw]
    return acc, np.array(time_dt), time_raw
