"""
Multi-Modal Telemetry Simulator (Temperature + Light/LDR).
Simulates real-world second-order environmental dynamics, target tracking,
and interactive presentation demo triggers.

Demo Controls:
- Press '1': Trigger Thermal Spike in Darkness (PyTorch Deep Autoencoder Anomaly)
- Press '2': Trigger Intense Solar Light Surge (XGBoost Preemptive Cooling)
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
    from src.config import SERVER_URL, TARGET_URL, DEMO_URL
except ImportError:
    from config import SERVER_URL, TARGET_URL, DEMO_URL

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


DEMO_BANNERS = {
    "DARK_HEAT_SPIKE": ">>> [DEMO TRIGGER]: Thermal Surge in Total Darkness (Fire/Fault Hazard)! <<<",
    "SOLAR_SURGE": ">>> [DEMO TRIGGER]: Intense Solar Light Exposure (Simulating Sun Heat Gain)! <<<",
    "LDR_FAULT": ">>> [DEMO TRIGGER]: LDR Sensor Disconnect (Stuck at 0%)! <<<",
    "NORMAL": ">>> [DEMO RESET]: Returned to Normal Operation. <<<",
}


def run_simulator(server_url=SERVER_URL, target_url=TARGET_URL, demo_url=DEMO_URL):
    print("=" * 70)
    print("  Multi-Modal Telemetry Simulator (Temperature + Light/LDR)  ")
    print("=" * 70)
    print(f"Target Server: {server_url}")
    print("\nInteractive Presentation Triggers:")
    print(" [1] -> Thermal Spike in Darkness (PyTorch Deep Autoencoder Anomaly)")
    print(" [2] -> Intense Solar Light Surge (XGBoost Preemptive Cooling)")
    print(" [3] -> LDR Sensor Disconnect / Fault")
    print(" [4] -> Shift Target Setpoint to 29.5C (Test Heating Pursuit)")
    print(" [5] -> Shift Target Setpoint to 26.0C (Test Cooling Pursuit)")
    print(" [0] -> Reset to Normal Dynamics & Target 27.0C")
    print("=" * 70 + "\n")

    temp = 25.4
    thermal_vel = 0.0
    light = 55.0
    led_1 = 0
    led_2 = 0
    tripped = False
    injection_mode = "NORMAL"

    def update_server_target(new_target_temp, new_target_light=50.0):
        try:
            requests.post(target_url, json={
                "target_temp": new_target_temp,
                "target_light": new_target_light
            }, timeout=2)
            print(f"\n>>> [SETPOINT CHANGED] Server Target updated to {new_target_temp}C! <<<\n")
        except Exception as e:
            print(f"[WARN] Could not update server target: {e}")

    def set_injection_mode(mode):
        nonlocal injection_mode
        if mode != injection_mode:
            injection_mode = mode
            print("\n" + DEMO_BANNERS.get(mode, f">>> [DEMO]: {mode} <<<") + "\n")

    def request_demo_mode(mode):
        try:
            requests.post(demo_url, json={"mode": mode}, timeout=2)
        except Exception as e:
            print(f"[WARN] Could not reach demo endpoint, applying locally: {e}")
        set_injection_mode(mode)

    try:
        step = 0
        while True:
            step += 1
            key = check_key()
            if key == '1':
                request_demo_mode("DARK_HEAT_SPIKE")
            elif key == '2':
                request_demo_mode("SOLAR_SURGE")
            elif key == '3':
                request_demo_mode("LDR_FAULT")
            elif key == '4':
                request_demo_mode("NORMAL")
                update_server_target(29.5)
            elif key == '5':
                request_demo_mode("NORMAL")
                update_server_target(26.0)
            elif key == '0':
                request_demo_mode("NORMAL")
                update_server_target(27.0)

            # Environmental physics simulation (with realistic second-order thermal inertia)
            if injection_mode == "DARK_HEAT_SPIKE":
                light = max(2.0, min(7.5, 4.5 + random.uniform(-1.0, 1.0)))
                thermal_vel = random.uniform(0.55, 0.90)
                temp += thermal_vel
            elif injection_mode == "SOLAR_SURGE":
                light = min(98.5, light + random.uniform(5.0, 10.0))
                thermal_vel = random.uniform(0.35, 0.60)
                temp += thermal_vel
            elif injection_mode == "LDR_FAULT":
                light = 0.0
                thermal_vel = 0.75 * thermal_vel + random.uniform(-0.02, 0.02)
                temp += thermal_vel
            else:
                # Normal environmental cycle: gentle mean-reverting daylight walk (18% - 88%)
                light += 0.08 * (55.0 - light) + random.uniform(-2.2, 2.2)
                light = max(18.0, min(88.0, light))

                solar_flux = 0.04 * (light - 50.0) / 50.0
                ambient_pull = -0.030 * (temp - 23.0)
                heater_drive = 0.40 if led_1 == 1 else 0.0
                thermal_vel = 0.78 * thermal_vel + 0.22 * (heater_drive + ambient_pull + solar_flux)
                temp += thermal_vel + random.uniform(-0.015, 0.015)

            temp = round(max(20.0, min(48.0, temp)), 2)
            light = round(max(0.0, min(100.0, light)), 1)

            payload = {
                "timestamp": int(time.time() * 1000),
                "temperature": temp,
                "light": light,
                "source": "simulator"
            }

            try:
                res = requests.post(server_url, json=payload, timeout=2)
                if res.status_code == 200:
                    body = res.json()
                    ctrl = body.get("control_signal", {})
                    insights = body.get("ai_insights", {})

                    server_mode = body.get("demo_mode")
                    if server_mode:
                        set_injection_mode(server_mode)

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
                    rec = rec.replace("°", "").encode("ascii", "ignore").decode().strip()

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
                print(f"[ERROR] Could not reach central server: {e}")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nSimulator stopped.")


if __name__ == "__main__":
    run_simulator()
