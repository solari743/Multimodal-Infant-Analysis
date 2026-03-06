import h5py
import numpy as np
import datetime


def load_accelerometer_data(filename, sensor_id):
    with h5py.File(filename, 'r') as f:
        if sensor_id not in f['Sensors']:
            raise KeyError(
                f"Sensor ID {sensor_id} not found. "
                f"Available IDs: {list(f['Sensors'].keys())}"
            )

        base_path = f"Sensors/{sensor_id}"

        if 'Accelerometer' not in f[base_path] or 'Time' not in f[base_path]:
            raise KeyError(f"Missing Accelerometer or Time data for Sensor {sensor_id}")

        acc_data = np.array(f[f"{base_path}/Accelerometer"][:],dtype=np.float64)

        time_raw = np.array(f[f"{base_path}/Time"][:],dtype=np.float64)

        time_dt = [
            datetime.datetime.fromtimestamp(t * 1e-6)
            for t in time_raw
        ]

    return acc_data, time_dt