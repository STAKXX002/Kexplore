"""
Multi-Modal Historical Training Dataset Generator.
Simulates thermodynamic heating/cooling, solar-thermal cross-coupling,
daylight illumination cycles (LDR), and multi-modal anomaly modes.
"""

import numpy as np
import pandas as pd
import time

def generate_telemetry_dataset(num_samples=4200, output_file="historical_training_data.csv"):
    """
    Generates synthetic multi-modal telemetry data (Temperature + Light/LDR).
    """
    np.random.seed(42)
    
    ambient_temp = 24.5
    heating_target = 35.5
    
    temp = 26.5
    light = 55.0  # Percentage 0 - 100%
    
    led_1 = 0     # Thermal Actuator / Cooling (1 = ON/Heating, 0 = OFF/Cooling)
    led_2 = 0     # Smart Illumination (1 = ON/Dark, 0 = OFF/Bright)
    tripped = 0
    
    records = []
    current_time = int(time.time() * 1000) - (num_samples * 2000)
    
    # Anomaly intervals
    # (start, end, anomaly_type)
    anomaly_intervals = [
        (650, 690, "thermal_spike_in_dark"),   # Fire/short circuit in darkness
        (1500, 1540, "ldr_sensor_failure"),     # LDR pin disconnected/stuck at 0%
        (2400, 2430, "solar_thermal_overheat"), # Intense solar heat wave
        (3300, 3340, "thermal_spike_in_dark"),
        (3850, 3880, "ldr_sensor_failure")
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
        
        # Diurnal solar cycle: sinusoidal variation simulating daylight (20% to 90%)
        # period of ~1000 steps
        solar_base = 55.0 + 30.0 * np.sin(2 * np.pi * i / 1000.0)
        
        if a_type == "thermal_spike_in_dark":
            # Pitch dark room, but temperature surges abnormally!
            light = max(2.0, min(8.0, 5.0 + np.random.normal(0, 1.0)))
            temp += np.random.uniform(0.4, 0.7)
            
        elif a_type == "ldr_sensor_failure":
            # LDR sensor failure: drops to 0 or spikes to 100 while temp continues normal solar walk
            light = 0.0 if i % 2 == 0 else 100.0
            temp += np.random.normal(0, 0.05)
            
        elif a_type == "solar_thermal_overheat":
            # Extreme solar radiation causing thermal runaway
            light = np.clip(95.0 + np.random.normal(0, 1.5), 85.0, 100.0)
            temp += np.random.uniform(0.3, 0.6)
            
        else:
            # Normal environmental dynamics
            light = np.clip(solar_base + np.random.normal(0, 1.5), 5.0, 95.0)
            
            # Solar radiation contributes to ambient heat gain
            solar_heat_gain = 0.02 * (light - 50.0) / 50.0
            
            # Classical safety rules & thermostat hysteresis
            if temp >= 32.0:
                tripped = 1
                led_1 = 0
            elif tripped and temp < 28.5:
                tripped = 0
                led_1 = 1
            elif not tripped:
                if temp <= 27.2:
                    led_1 = 1
                elif temp >= 30.8:
                    led_1 = 0

            # Thermal update with thermal inertia & solar heat gain
            if led_1 == 1:
                temp += 0.075 * (heating_target - temp) + solar_heat_gain + np.random.normal(0, 0.03)
            else:
                temp -= 0.060 * (temp - ambient_temp) - solar_heat_gain + np.random.normal(0, 0.03)

        # Smart daylight harvesting illumination rule:
        # If natural light is below 35%, turn ON illumination LED 2
        led_2 = 1 if (light < 35.0 or is_anomaly == 1) else 0
        
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
        
    df = pd.DataFrame(records)
    
    # Target: temperature t+5 intervals ahead (~10 seconds in future)
    df["target_temp_next5"] = df["temperature"].shift(-5)
    df = df.dropna().reset_index(drop=True)
    
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} multi-modal samples saved to '{output_file}'.")
    print(f"Normal: {sum(df['is_anomaly'] == 0)}, Anomaly: {sum(df['is_anomaly'] == 1)}")
    return df

if __name__ == "__main__":
    generate_telemetry_dataset()
