import os

from core.h5_loader import load_accelerometer_data
from core.accel_magnitude import (
    compute_acceleration_magnitude,
    compute_epoch_features
)
from plots.accel_magnitude_plot import plot_acceleration_magnitude


def main():

    filename = r'path/to/your/file.h5'
    target_ID = "Your_Sensor_ID"

    if not os.path.exists(filename):
        print(f"File '{filename}' not found.")
        return

    acc_data, time_dt = load_accelerometer_data(
        filename,
        target_ID
    )

    df = compute_acceleration_magnitude(
        acc_data,
        time_dt
    )

    epoch_features = compute_epoch_features(df)

    print("\n--- Acceleration Magnitude Statistics ---")
    print(epoch_features.head())

    output_csv_name = (f"{os.path.splitext(filename)[0]}""_acc_magnitude_per_epoch.csv")

    epoch_features.to_csv(output_csv_name)

    print(f"\nSaved epoch statistics to {output_csv_name}")

    output_plot_name = (f"{os.path.splitext(filename)[0]}" "_acceleration_magnitude_plot.png")

    plot_acceleration_magnitude(epoch_features,output_plot_name)


if __name__ == "__main__":
    main()