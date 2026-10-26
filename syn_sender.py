import time
import random
import requests

# Set your Flask server URL 
# (Use 'http://localhost:5000/telemetry' if running on the same PC, 
# or 'http://<HOST_IP>:5000/telemetry' if running on a separate network device)
SERVER_URL = "http://127.0.0.1:5000/telemetry"

print("=== Starting Synthetic Telemetry Simulator ===")
print(f"Target Server: {SERVER_URL}\n")

# Simulated state variables
current_temp = 27.5
current_hum = 50.0
is_auto = True
tripped = False

try:
    while True:
        # Simulate realistic temperature drift (random walk)
        current_temp += random.uniform(-0.4, 0.5)
        current_hum += random.uniform(-1.0, 1.0)
        
        # Keep bounds realistic
        current_temp = max(20.0, min(35.0, current_temp))
        current_hum = max(30.0, __import__('builtins').min(80.0, current_hum))

        # Check automated logic rules
        if current_temp >= 32.0:
            tripped = True
            load_state = "OFF"
        elif tripped and current_temp < 29.0:
            tripped = False # Recovered after cooling down
            load_state = "ON" if is_auto else "OFF"
        elif not tripped:
            load_state = "ON" if (is_auto and current_temp >= 28.0) else "OFF"

        # Construct payload matching ESP32 structure
        payload = {
            "timestamp": int(time.time() * 1000),
            "temperature": round(current_temp, 1),
            "humidity": round(current_hum, 1),
            "load_state": load_state,
            "mode": "AUTO" if is_auto else "MANUAL",
            "tripped": tripped
        }

        try:
            response = requests.post(SERVER_URL, json=payload, timeout=2)
            if response.status_code == 200:
                print(f"[SENT] Temp: {payload['temperature']}°C | Hum: {payload['humidity']}% | Load: {load_state} | Tripped: {tripped}")
            else:
                print(f"[WARN] Server responded with code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Could not reach server: {e}")

        # Wait 2 seconds to mimic the DHT22 interval
        time.sleep(2)

except KeyboardInterrupt:
    print("\nSimulator stopped by user.")
