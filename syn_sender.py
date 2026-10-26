"""
Synthetic Dual-Sensor Telemetry Simulator with Interactive Anomaly Injection.
Simulates 2 sensor nodes sending telemetry to receiver.py and acting on
the returned AI control signals.

Interactive Demo Controls:
- Press '1': Trigger rapid Heat Spike on Sensor 1 (to demo AI Predictive Cooling)
- Press '2': Inject Sensor Divergence / Drift (to demo Isolation Forest Anomaly Detection)
- Press '0': Return to Normal Thermodynamic Cycling
- Press 'Ctrl+C': Exit simulator
"""

import time
import random
import requests
import sys

# Windows non-blocking keypress check
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

print("=" * 65)
print("     Synthetic Dual-Sensor Telemetry & Actuation Simulator     ")
print("=" * 65)
print(f"Target Server: {SERVER_URL}")
print("\nInteractive Presentation Triggers:")
print(" [1] -> Inject Heat Spike (Showcase AI Predictive Forecast)")
print(" [2] -> Inject Sensor Drift / Fault (Showcase Anomaly Detection)")
print(" [0] -> Reset to Normal Cycling")
print("=" * 65 + "\n")

# State variables
s1_temp = 27.2
s2_temp = 27.0
s1_hum = 52.0
s2_hum = 51.5

load_1 = 0
load_2 = 0
tripped = False

# Active demo injection mode
injection_mode = "NORMAL"

try:
    step = 0
    while True:
        step += 1
        
        # Check for interactive keyboard input
        key = check_key()
        if key == '1':
            injection_mode = "HEAT_SPIKE"
            print("\n>>> [DEMO TRIGGER ACTIVATED]: Rapid Heat Spike Injected on Sensor 1! <<<\n")
        elif key == '2':
            injection_mode = "SENSOR_DRIFT"
            print("\n>>> [DEMO TRIGGER ACTIVATED]: Cross-Sensor Drift Injected! <<<\n")
        elif key == '0':
            injection_mode = "NORMAL"
            print("\n>>> [DEMO RESET]: Returned to Normal Operation. <<<\n")

        # Simulate physics based on current load and injection
        if injection_mode == "HEAT_SPIKE":
            s1_temp += random.uniform(0.6, 1.1)
            s2_temp += random.uniform(0.2, 0.4)
        elif injection_mode == "SENSOR_DRIFT":
            s1_temp += random.uniform(-0.1, 0.2)
            s2_temp += random.uniform(0.7, 1.2)  # Sensor 2 drifts significantly away
        else:
            # Normal thermodynamics:
            # If load_1 is ON, heat rises; if OFF, cools toward 25°C ambient
            if load_1 == 1:
                s1_temp += random.uniform(0.15, 0.35)
                s2_temp += random.uniform(0.12, 0.30)
            else:
                s1_temp -= random.uniform(0.10, 0.25) if s1_temp > 25.5 else random.uniform(-0.1, 0.1)
                s2_temp -= random.uniform(0.08, 0.22) if s2_temp > 25.5 else random.uniform(-0.1, 0.1)

        # Humidity updates
        s1_hum += random.uniform(-0.5, 0.5)
        s2_hum += random.uniform(-0.5, 0.5)

        # Keep bounded
        s1_temp = round(max(20.0, min(48.0, s1_temp)), 2)
        s2_temp = round(max(20.0, min(48.0, s2_temp)), 2)
        s1_hum = round(max(30.0, min(80.0, s1_hum)), 1)
        s2_hum = round(max(30.0, min(80.0, s2_hum)), 1)

        payload = {
            "timestamp": int(time.time() * 1000),
            "sensor_1": s1_temp,
            "sensor_2": s2_temp,
            "humidity_1": s1_hum,
            "humidity_2": s2_hum
        }

        try:
            res = requests.post(SERVER_URL, json=payload, timeout=2)
            if res.status_code == 200:
                body = res.json()
                ctrl = body.get("control_signal", {})
                insights = body.get("ai_insights", {})

                # Update local simulated actuator state based on server's control decision
                load_1 = ctrl.get("led_1", 0)
                load_2 = ctrl.get("led_2", 0)
                tripped = ctrl.get("tripped", False)
                mode = ctrl.get("mode", "UNKNOWN")

                pred_t = insights.get("predicted_temp", s1_temp)
                is_anom = insights.get("is_anomaly", False)
                anom_reason = insights.get("anomaly_reason", "Normal")

                status_flag = "⚠️ ANOMALY" if is_anom else ("🛑 TRIPPED" if tripped else "✅ OPTIMAL")
                print(f"[SENT] S1: {s1_temp}°C | S2: {s2_temp}°C | Mode: {injection_mode}")
                print(f"       ==> AI Pred(t+5): {pred_t}°C | Status: {status_flag}")
                print(f"       ==> Server Control: LED 1: {'ON' if load_1 else 'OFF'} | LED 2: {'ON' if load_2 else 'OFF'} | Policy: {mode}\n")

            else:
                print(f"[WARN] Server responded with code: {res.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Could not reach receiver.py at {SERVER_URL}: {e}")

        # Sleep interval (2 seconds between telemetry frames)
        time.sleep(2)

except KeyboardInterrupt:
    print("\nSimulator stopped by user.")
