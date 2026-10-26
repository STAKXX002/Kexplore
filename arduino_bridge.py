"""
Modular Arduino Hardware Communication Bridge.
Isolates all physical serial communication with the Arduino board.
Forwards sensor readings to receiver.py and sends back LED control signals.

This file is completely modular: modifications to the AI models,
dashboard, or server will NEVER break this hardware communication.
"""

import time
import json
import requests
import argparse
import sys

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[ERROR] pyserial is not installed. Run: pip install pyserial")
    sys.exit(1)

# Default configuration
DEFAULT_SERVER_URL = "http://127.0.0.1:5000/telemetry"
DEFAULT_BAUD = 9600

def list_available_ports():
    """Returns a list of available serial COM ports."""
    ports = list(serial.tools.list_ports.comports())
    return [p.device for p in ports]

def auto_detect_port():
    """Tries to find an Arduino or active COM port."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    for p in ports:
        # Check description for Arduino or CH340 or USB Serial
        desc = (p.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "usb-serial" in desc or "usb serial" in desc:
            return p.device
    # Default to first available port
    return ports[0].device

def parse_sensor_line(line_str):
    """
    Robust sensor reading parser. Supports:
    1. JSON format: {"sensor_1": 28.5, "sensor_2": 27.8}
    2. Comma-separated: 28.5, 27.8
    3. Key-value: S1=28.5, S2=27.8
    """
    line = line_str.strip()
    if not line:
        return None

    # Case 1: JSON
    if line.startswith("{") and line.endswith("}"):
        try:
            data = json.loads(line)
            return {
                "sensor_1": float(data.get("sensor_1", data.get("s1", 25.0))),
                "sensor_2": float(data.get("sensor_2", data.get("s2", 25.0)))
            }
        except Exception:
            pass

    # Case 2: Key-value e.g. S1:28.4,S2:27.9 or S1=28.4,S2=27.9
    if "=" in line or ":" in line:
        try:
            parts = line.replace(";", ",").split(",")
            readings = {}
            for part in parts:
                sep = "=" if "=" in part else ":"
                k, v = part.split(sep, 1)
                k_clean = k.strip().lower()
                if "1" in k_clean:
                    readings["sensor_1"] = float(v.strip())
                elif "2" in k_clean:
                    readings["sensor_2"] = float(v.strip())
            if "sensor_1" in readings and "sensor_2" in readings:
                return readings
        except Exception:
            pass

    # Case 3: Comma separated e.g. "28.5, 27.9"
    if "," in line:
        try:
            parts = line.split(",")
            return {
                "sensor_1": float(parts[0].strip()),
                "sensor_2": float(parts[1].strip())
            }
        except Exception:
            pass

    return None

def run_bridge(port=None, baud=DEFAULT_BAUD, server_url=DEFAULT_SERVER_URL):
    print("=" * 65)
    print("      Modular Arduino Hardware Serial <-> HTTP Bridge        ")
    print("=" * 65)
    
    if port is None:
        port = auto_detect_port()

    if not port:
        available = list_available_ports()
        print("[WARNING] No active COM port detected!")
        if available:
            print(f"Available ports: {', '.join(available)}")
        else:
            print("No serial devices found. Please plug in your Arduino USB cable.")
        print("You can also run synthetic testing via: python syn_sender.py")
        print("Exiting bridge gracefully.\n")
        return

    print(f"[Bridge Config] Target Serial Port: {port} @ {baud} baud")
    print(f"[Bridge Config] Target Server URL:  {server_url}\n")
    print(f"Attempting connection to Arduino on {port}...")

    try:
        ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)  # Wait for Arduino auto-reset after serial connect
        print(f" Connected to Arduino on {port}!")
        print("Streaming sensor readings to AI server... Press Ctrl+C to stop.\n")
    except Exception as e:
        print(f"[ERROR] Could not open serial port {port}: {e}")
        print("Please check your USB connection or specify port using: python arduino_bridge.py --port COMx")
        return

    try:
        while True:
            if ser.in_waiting > 0:
                raw_bytes = ser.readline()
                try:
                    line = raw_bytes.decode('utf-8', errors='ignore').strip()
                except Exception:
                    continue

                if not line:
                    continue

                readings = parse_sensor_line(line)
                if not readings:
                    # Echo raw debug lines from Arduino if any
                    print(f"[Arduino Raw] {line}")
                    continue

                s1 = readings["sensor_1"]
                s2 = readings["sensor_2"]

                payload = {
                    "timestamp": int(time.time() * 1000),
                    "sensor_1": s1,
                    "sensor_2": s2,
                    "humidity_1": 50.0,
                    "humidity_2": 50.0
                }

                # Forward to AI Server
                try:
                    res = requests.post(server_url, json=payload, timeout=2)
                    if res.status_code == 200:
                        res_data = res.json()
                        ctrl = res_data.get("control_signal", {})
                        insights = res_data.get("ai_insights", {})

                        led_1 = ctrl.get("led_1", 0)
                        led_2 = ctrl.get("led_2", 0)
                        mode = ctrl.get("mode", "UNKNOWN")
                        pred_t = insights.get("predicted_temp", s1)
                        is_anom = insights.get("is_anomaly", False)

                        print(f"[TELEMETRY] S1: {s1}°C | S2: {s2}°C  ==>  [AI Pred: {pred_t}°C | Anomaly: {is_anom}]")
                        print(f"[ACTUATE]   Sending to Arduino -> LED 1: {led_1} | LED 2: {led_2} | Mode: {mode}\n")

                        # Send control command back to Arduino
                        cmd = f'{{"led_1":{led_1},"led_2":{led_2}}}\n'
                        ser.write(cmd.encode('utf-8'))

                    else:
                        print(f"[WARN] Server responded with HTTP {res.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"[WARN] Could not reach AI server at {server_url}: {e}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[Bridge] Stopping hardware bridge. Turning off LEDs...")
        try:
            ser.write(b'{"led_1":0,"led_2":0}\n')
            time.sleep(0.5)
            ser.close()
        except Exception:
            pass
        print("[Bridge] Hardware connection closed cleanly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arduino Serial to AI Server Bridge")
    parser.add_argument("--port", type=str, default=None, help="Serial COM port (e.g., COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=9600, help="Baud rate (default: 9600)")
    parser.add_argument("--url", type=str, default=DEFAULT_SERVER_URL, help="Server telemetry URL")
    args = parser.parse_args()

    run_bridge(port=args.port, baud=args.baud, server_url=args.url)
