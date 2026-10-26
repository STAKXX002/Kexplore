"""
Historical Training Dataset Generator for Dual-Sensor IoT Telemetry.
Simulates realistic thermodynamic heating/cooling cycles, dual-sensor correlation,
and various industrial anomaly modes (spikes, sensor drift, disconnects).
"""

import numpy as np
import pandas as pd
import time

def generate_telemetry_dataset(num_samples=4000, output_file="historical_training_data.csv"):
    """
    Generates synthetic telemetry data representing ~2.2 hours of 2-second interval readings.
    """
    np.random.seed(42)
    
    ambient_temp = 25.0
    heating_target = 35.0
    
    s1_temp = 26.5
    s2_temp = 26.2
    s1_hum = 55.0
    s2_hum = 54.0
    
    load_state = 0
    tripped = 0
    
    records = []
    current_time = int(time.time() * 1000) - (num_samples * 2000)
    
    # Anomaly intervals distributed across training and testing splits
    anomaly_intervals = [
        (600, 630, "thermal_runaway"),
        (1400, 1440, "sensor_drift"),
        (2200, 2220, "sensor_glitch"),
        (3200, 3230, "thermal_runaway"),
        (3700, 3730, "sensor_drift")
    ]
    
    def get_anomaly_type(idx):
        for start, end, a_type in anomaly_intervals:
            if start <= idx <= end:
                return a_type
        return None

    for i in range(num_samples):
        a_type = get_anomaly_type(i)
        anomaly = 1 if a_type is not None else 0
        anomaly_name = a_type if a_type is not None else "normal"
        
        if a_type == "thermal_runaway":
            s1_temp += np.random.uniform(0.35, 0.65)
            s2_temp += np.random.uniform(0.30, 0.60)
        elif a_type == "sensor_drift":
            s1_temp += np.random.uniform(-0.05, 0.1)
            s2_temp += np.random.uniform(0.3, 0.5)  # Sensor 2 drifts upwards
        elif a_type == "sensor_glitch":
            s1_temp = np.random.uniform(46.0, 52.0)
            s2_temp = s2_temp + np.random.uniform(-0.1, 0.1)
        else:
            # Normal thermodynamic cycling
            if s1_temp >= 32.0 or s2_temp >= 32.0:
                tripped = 1
                load_state = 0
            elif tripped and (s1_temp < 28.5 and s2_temp < 28.5):
                tripped = 0
                load_state = 1
            elif not tripped:
                if s1_temp <= 27.5:
                    load_state = 1
                elif s1_temp >= 31.0:
                    load_state = 0

            # Thermal updates
            if load_state == 1:
                # Heating up smoothly with thermal inertia
                s1_temp += 0.08 * (heating_target - s1_temp) + np.random.normal(0, 0.04)
                s2_temp += 0.075 * (heating_target - s2_temp) + np.random.normal(0, 0.04)
            else:
                # Cooling down toward ambient
                s1_temp -= 0.06 * (s1_temp - ambient_temp) + np.random.normal(0, 0.04)
                s2_temp -= 0.055 * (s2_temp - ambient_temp) + np.random.normal(0, 0.04)

        # Humidity updates (inverse relation with temperature)
        s1_hum = np.clip(65.0 - (s1_temp - 24.0) * 1.5 + np.random.normal(0, 0.2), 30.0, 85.0)
        s2_hum = np.clip(64.0 - (s2_temp - 24.0) * 1.5 + np.random.normal(0, 0.2), 30.0, 85.0)
        
        records.append({
            "timestamp": current_time + (i * 2000),
            "sensor_1_temp": round(float(s1_temp), 2),
            "sensor_1_hum": round(float(s1_hum), 2),
            "sensor_2_temp": round(float(s2_temp), 2),
            "sensor_2_hum": round(float(s2_hum), 2),
            "load_1_state": load_state,
            "load_2_state": 1 if s2_temp >= 28.0 and not tripped else 0,
            "tripped": tripped,
            "is_anomaly": anomaly,
            "anomaly_type": anomaly_name
        })
        
    df = pd.DataFrame(records)
    
    # Target variables: t+5 future temperature (approx 10s ahead)
    df["target_s1_temp_next5"] = df["sensor_1_temp"].shift(-5)
    df["target_s2_temp_next5"] = df["sensor_2_temp"].shift(-5)
    
    df = df.dropna().reset_index(drop=True)
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} samples saved to '{output_file}'.")
    print(f"Normal samples: {sum(df['is_anomaly'] == 0)}, Anomaly samples: {sum(df['is_anomaly'] == 1)}")
    return df

if __name__ == "__main__":
    generate_telemetry_dataset()
