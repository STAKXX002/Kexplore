"""
Multi-Modal Historical Training Dataset Generator.
Simulates second-order thermodynamic heating/cooling inertia, solar-thermal
cross-coupling, daylight illumination cycles (LDR), multi-setpoint regulation
across 21°C-34°C, and multi-modal anomaly modes.
"""

import numpy as np
import pandas as pd
import time

try:
    from src.config import TRAINING_DATA_FILE
except ImportError:
    from config import TRAINING_DATA_FILE


def generate_telemetry_dataset(num_samples=6000, output_file=None):
    """
    Generates synthetic multi-modal telemetry data (Temperature + Light/LDR)
    spanning the full 21°C - 35°C operating envelope so tree-based and neural
    models generalize across cold starts, coasting, warm setpoints, and faults.
    """
    if output_file is None:
        output_file = TRAINING_DATA_FILE

    np.random.seed(42)

    ambient_temp = 23.0
    temp = 25.0
    thermal_vel = 0.0
    light = 55.0  # Percentage 0 - 100%

    led_1 = 0     # Thermal Actuator (1 = ON/Heating, 0 = OFF/Cooling)
    led_2 = 0     # Smart Illumination (1 = ON/Dark, 0 = OFF/Bright)
    tripped = 0

    records = []
    current_time = int(time.time() * 1000) - (num_samples * 2000)

    # Rotating setpoints across the full operating range + open-loop holds
    setpoint_schedule = [27.0, 29.5, 24.5, 30.5, 23.5, 28.0, 25.5, 31.0, 22.5, 27.5]
    hold_counter = 0

    # Anomaly intervals: (start, end, anomaly_type)
    anomaly_intervals = [
        (650, 695, "thermal_spike_in_dark"),     # Fire/short circuit in darkness
        (1500, 1545, "ldr_sensor_failure"),      # LDR pin disconnected/stuck at 0/100%
        (2400, 2440, "solar_thermal_overheat"),  # Intense solar heat wave
        (3300, 3345, "thermal_spike_in_dark"),
        (4200, 4245, "ldr_sensor_failure"),
        (5100, 5140, "solar_thermal_overheat"),
    ]

    def get_anomaly_type(idx):
        for start, end, a_type in anomaly_intervals:
            if start <= idx <= end:
                return a_type
        return None

    for i in range(num_samples):
        a_type = get_anomaly_type(i)
        is_anomaly = 1 if a_type is not None else 0
        anomaly_name = a_type if a_type is not None else "normal"

        # Diurnal solar cycle: sinusoidal variation simulating daylight (20% to 88%)
        solar_base = 54.0 + 30.0 * np.sin(2 * np.pi * i / 900.0)
        active_target = setpoint_schedule[(i // 300) % len(setpoint_schedule)]

        if a_type == "thermal_spike_in_dark":
            # Pitch dark room, but temperature surges abnormally!
            light = float(np.clip(4.0 + np.random.normal(0, 1.0), 1.5, 7.5))
            led_1 = 0
            led_2 = 1
            records.append({
                "timestamp": current_time + (i * 2000),
                "temperature": round(float(temp), 2),
                "light": round(float(light), 2),
                "led_1": led_1,
                "led_2": led_2,
                "tripped": tripped,
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_name
            })
            thermal_vel = np.random.uniform(0.55, 0.90)
            temp = float(np.clip(temp + thermal_vel, 20.5, 42.0))

        elif a_type == "ldr_sensor_failure":
            # LDR sensor failure: drops to 0 or spikes to 100
            light = 0.0 if (i // 4) % 2 == 0 else 100.0
            led_2 = 1
            records.append({
                "timestamp": current_time + (i * 2000),
                "temperature": round(float(temp), 2),
                "light": round(float(light), 2),
                "led_1": led_1,
                "led_2": led_2,
                "tripped": tripped,
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_name
            })
            thermal_vel = 0.75 * thermal_vel + np.random.normal(0, 0.02)
            temp = float(np.clip(temp + thermal_vel, 20.5, 36.0))

        elif a_type == "solar_thermal_overheat":
            # Extreme solar radiation causing thermal runaway above safety limits
            light = float(np.clip(97.0 + np.random.normal(0, 1.0), 94.0, 100.0))
            led_1 = 0
            led_2 = 1
            records.append({
                "timestamp": current_time + (i * 2000),
                "temperature": round(float(temp), 2),
                "light": round(float(light), 2),
                "led_1": led_1,
                "led_2": led_2,
                "tripped": tripped,
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_name
            })
            thermal_vel = np.random.uniform(0.45, 0.75)
            temp = float(np.clip(temp + thermal_vel, 20.5, 42.0))

        else:
            # Normal environmental dynamics
            light = float(np.clip(solar_base + np.random.normal(0, 2.0), 15.0, 90.0))
            solar_flux = 0.04 * (light - 50.0) / 50.0

            # Safety ceiling & multi-regime excitation (closed-loop + random holds
            # so both led_1=0 and led_1=1 are observed across all temperatures)
            if temp >= 32.0:
                tripped = 1
                led_1 = 0
                hold_counter = 0
            elif tripped and temp < 28.5:
                tripped = 0

            if not tripped:
                if hold_counter > 0:
                    hold_counter -= 1
                elif np.random.rand() < 0.08:
                    # 8-15 step exploratory hold so XGBoost observes counterfactuals
                    # (both heating and cooling) at every temperature and velocity
                    led_1 = int(np.random.choice([0, 1]))
                    hold_counter = int(np.random.randint(8, 16))
                else:
                    if temp < active_target - 0.25:
                        led_1 = 1
                    elif temp > active_target + 0.25:
                        led_1 = 0

            led_2 = 1 if light < 50.0 else 0

            # Record state (T_t, light_t) + chosen action u_t BEFORE stepping physics
            records.append({
                "timestamp": current_time + (i * 2000),
                "temperature": round(float(temp), 2),
                "light": round(float(light), 2),
                "led_1": led_1,
                "led_2": led_2,
                "tripped": tripped,
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_name
            })

            # Second-order thermal inertia (heater stored heat + Newton cooling)
            ambient_pull = -0.030 * (temp - ambient_temp)
            heater_drive = 0.40 if led_1 == 1 else 0.0
            thermal_vel = 0.78 * thermal_vel + 0.22 * (heater_drive + ambient_pull + solar_flux)
            temp += thermal_vel + np.random.normal(0, 0.015)
            temp = float(np.clip(temp, 20.5, 36.0))

        # Smart daylight harvesting illumination rule
        led_2 = 1 if (light < 50.0 or is_anomaly == 1) else 0

    df = pd.DataFrame(records)

    # Target: temperature t+5 intervals ahead (~10 seconds in future)
    df["target_temp_next5"] = df["temperature"].shift(-5)
    df = df.dropna().reset_index(drop=True)

    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} multi-modal samples saved to '{output_file}'.")
    print(f"Temp range: [{df['temperature'].min():.1f}C, {df['temperature'].max():.1f}C] | "
          f"Normal: {sum(df['is_anomaly'] == 0)}, Anomaly: {sum(df['is_anomaly'] == 1)}")
    return df


if __name__ == "__main__":
    generate_telemetry_dataset()
