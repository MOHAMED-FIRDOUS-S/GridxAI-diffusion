
import numpy as np
import pandas as pd
from pathlib import Path


def generate_smart_grid_data(
    num_nodes=10,
    readings_per_node=100,
    random_state=42
):
    rng = np.random.default_rng(random_state)
    records = []

    start_time = pd.Timestamp("2026-01-01 00:00:00")

    for node in range(num_nodes):
        base_voltage = rng.uniform(220, 240)
        base_current = rng.uniform(5, 20)
        base_frequency = 50.0
        base_power = base_voltage * base_current

        for t in range(readings_per_node):
            voltage = base_voltage + rng.normal(0, 1.5)
            current = base_current + rng.normal(0, 0.5)
            frequency = base_frequency + rng.normal(0, 0.03)
            power = voltage * current

            is_anomaly = 0
            anomaly_type = "normal"

            # Inject a simulated sensor manipulation event.
            if 60 <= t < 65 and node in (2, 5, 8):
                voltage += rng.choice([-35, 35])
                current *= rng.uniform(1.8, 2.2)
                is_anomaly = 1
                anomaly_type = "injected_sensor_anomaly"
                power = voltage * current

            records.append({
                "timestamp": start_time + pd.Timedelta(minutes=t),
                "node_id": f"Node_{node:02d}",
                "voltage": round(voltage, 3),
                "current": round(current, 3),
                "frequency": round(frequency, 4),
                "power": round(power, 3),
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_type
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    output_file = project_root / "data" / "smart_grid_data.csv"

    df = generate_smart_grid_data()
    df.to_csv(output_file, index=False)

    print("Smart-grid dataset generated successfully!")
    print(f"File saved at: {output_file}")
    print(f"Total readings: {len(df)}")
    print(f"Total grid nodes: {df['node_id'].nunique()}")
    print(f"Injected anomalies: {df['is_anomaly'].sum()}")
    print("\nFirst five records:")
    print(df.head().to_string(index=False))
