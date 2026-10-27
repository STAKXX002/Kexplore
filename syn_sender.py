"""
Multi-Modal Telemetry Simulator (Temperature + Light/LDR).
Simulates real-world environmental dynamics, target tracking, and interactive anomaly triggers.

Demo Controls:
- Press '1': Trigger Thermal Spike in Darkness (Demos PyTorch Deep Autoencoder Anomaly)
- Press '2': Trigger Intense Solar Light Surge (Demos XGBoost Preemptive Cooling)
- Press '3': Trigger LDR Light Sensor Disconnect Fault
- Press '4': Set Target Temp to 29.5C (Warm Setpoint Tracking)
- Press '5': Set Target Temp to 26.0C (Cool Setpoint Tracking)
- Press '0': Reset Target to 27.0C and Normal Environmental Dynamics
- Press 'Ctrl+C': Exit simulator
"""

import time
import random
import requests
import sys

try:
    import msvcrt
    def check_key():
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            try:
                return ch.decode('utf-8')
            except Exception:
                return ""
        return None
except ImportError:
    def check_key():
        return None

SERVER_URL = "http://127.0.0.1:5000/telemetry"
TARGET_URL = "http://127.0.0.1:5000/api/set-target"

print("=" * 70)
print("  Multi-Modal Telemetry Simulator (Temperature + Light/LDR)  ")
print("=" * 70)
print(f"Target Server: {SERVER_URL}")
print("\nInteractive Presentation Triggers:")
print(" [1] -> Thermal Spike in Darkness (PyTorch Deep Autoencoder Anomaly)")
print(" [2] -> Intense Solar Light Surge (XGBoost Preemptive Cooling)")
print(" [3] -> LDR Sensor Disconnect / Fault")
print(" [4] -> Shift Target Setpoint to 29.5C (Test Heating Pursuit)")
print(" [5] -> Shift Target Setpoint to 26.0C (Test Cooling Pursuit)")
print(" [0] -> Reset to Normal Dynamics & Target 27.0C")
print("=" * 70 + "\n")

temp = 26.5
light = 55.0
led_1 = 0
led_2 = 0
tripped = False
injection_mode = "NORMAL"

def update_server_target(new_target_temp, new_target_light=50.0):
    try:
        requests.post(TARGET_URL, json={
            "target_temp": new_target_temp,
            "target_light": new_target_light
        }, timeout=2)
        print(f"\n>>> [SETPOINT CHANGED] Server Target updated to {new_target_temp}C! <<<\n")
    except Exception as e:
        print(f"[WARN] Could not update server target: {e}")

try:
    step = 0
    while True:
        step += 1
        key = check_key()
        if key == '1':
            injection_mode = "DARK_HEAT_SPIKE"
            print("\n>>> [DEMO TRIGGER]: Thermal Surge in Total Darkness (Fire/Fault Hazard)! <<<\n")
        elif key == '2':
            injection_mode = "SOLAR_SURGE"
            print("\n>>> [DEMO TRIGGER]: Intense Solar Light Exposure (Simulating Sun Heat Gain)! <<<\n")
        elif key == '3':
            injection_mode = "LDR_FAULT"
            print("\n>>> [DEMO TRIGGER]: LDR Sensor Disconnect (Stuck at 0%)! <<<\n")
        elif key == '4':
            injection_mode = "NORMAL"
            update_server_target(29.5)
        elif key == '5':
            injection_mode = "NORMAL"
            update_server_target(26.0)
        elif key == '0':
            injection_mode = "NORMAL"
            update_server_target(27.0)
            print("\n>>> [DEMO RESET]: Returned to Normal Operation & Target 27.0C. <<<\n")

        # Environmental physics simulation
        if injection_mode == "DARK_HEAT_SPIKE":
            light = max(2.0, min(8.0, 5.0 + random.uniform(-1.0, 1.0)))
            temp += random.uniform(0.6, 1.1)
        elif injection_mode == "SOLAR_SURGE":
            light = min(98.0, light + random.uniform(5.0, 10.0))
            temp += random.uniform(0.3, 0.6)  # Solar heat flux
        elif injection_mode == "LDR_FAULT":
            light = 0.0
            temp += random.uniform(-0.1, 0.1)
        else:
            # Normal environmental cycle
            # Solar light walk
            light += random.uniform(-2.5, 2.5)
            light = max(10.0, min(90.0, light))
            
            solar_flux = 0.02 * (light - 50.0) / 50.0
            if led_1 == 1:
                # Heating element / thermal load active
                temp += random.uniform(0.15, 0.30) + solar_flux
            else:
                # Cooling / passive dissipation down toward 24.5C baseline
                if temp > 25.0:
                    temp -= random.uniform(0.12, 0.22)
                else:
                    temp += random.uniform(-0.05, 0.05)
                temp += solar_flux

        temp = round(max(20.0, min(48.0, temp)), 2)
        light = round(max(0.0, min(100.0, light)), 1)

        payload = {
            "timestamp": int(time.time() * 1000),
            "temperature": temp,
            "light": light
        }

        try:
            res = requests.post(SERVER_URL, json=payload, timeout=2)
            if res.status_code == 200:
                body = res.json()
                ctrl = body.get("control_signal", {})
                insights = body.get("ai_insights", {})

                led_1 = ctrl.get("led_1", 0)
                led_2 = ctrl.get("led_2", 0)
                tripped = ctrl.get("tripped", False)
                mode = ctrl.get("mode", "UNKNOWN")

                pred_t = insights.get("predicted_temp", temp)
                recon_err = insights.get("recon_loss", 0.0)
                is_anom = insights.get("is_anomaly", False)
                target_t = insights.get("target_temp", 27.0)
                temp_err = insights.get("temp_error", round(temp - target_t, 2))
                target_achieved = insights.get("target_achieved", False)
                rec = insights.get("recommendation", "")

                if is_anom:
                    status_flag = "[ANOMALY]"
                elif tripped:
                    status_flag = "[TRIPPED]"
                elif target_achieved:
                    status_flag = "[TARGET LOCKED]"
                else:
                    status_flag = "[TRACKING TARGET]"

                err_str = f"+{temp_err:.2f}" if temp_err >= 0 else f"{temp_err:.2f}"
                print(f"[SENT] Temp: {temp}C (Target: {target_t}C, Err: {err_str}C) | Light: {light}% | Status: {status_flag}")
                print(f"       ==> XGBoost Forecast(t+5): {pred_t}C | Autoencoder Loss: {recon_err}")
                print(f"       ==> Actuators: LED 1: {'ON' if led_1 else 'OFF'} | LED 2: {'ON' if led_2 else 'OFF'} | {rec}\n")

            else:
                print(f"[WARN] HTTP {res.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Could not reach receiver.py: {e}")

        time.sleep(2)

except KeyboardInterrupt:
    print("\nSimulator stopped.")
