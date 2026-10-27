"""
Modular Arduino Hardware Communication Bridge (Temperature + Light/LDR).
Isolates all physical serial communication with the Arduino board.
Forwards temperature and light/LDR readings to central controller and sends back
dual LED actuation signals (LED 1: Thermal Load, LED 2: Smart Illumination).
"""

import time
import json
import requests
import argparse
import sys

try:
    from src.config import SERVER_URL, DEFAULT_BAUD
except ImportError:
    from config import SERVER_URL, DEFAULT_BAUD

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[ERROR] pyserial is not installed. Run: pip install pyserial")
    sys.exit(1)


def list_available_ports():
    ports = list(serial.tools.list_ports.comports())
    return [p.device for p in ports]


def auto_detect_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    for p in ports:
        desc = (p.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "usb-serial" in desc or "usb serial" in desc:
            return p.device
    return ports[0].device


def parse_sensor_line(line_str):
    """
    Parses multi-modal telemetry lines:
    1. JSON: {"temperature": 28.5, "light": 65.0} or {"s1": 28.5, "s2": 65.0}
    2. Comma-separated: "28.5, 65.0"
    3. Key-value: "temp=28.5, light=65.0"
    """
    line = line_str.strip()
    if not line:
        return None

    # Case 1: JSON
    if line.startswith("{") and line.endswith("}"):
        try:
            data = json.loads(line)
            temp = data.get("temperature", data.get("temp", data.get("sensor_1", data.get("s1"))))
            light = data.get("light", data.get("ldr", data.get("sensor_2", data.get("s2"))))
            if temp is not None and light is not None:
                return {"temperature": float(temp), "light": float(light)}
        except Exception:
            pass

    # Case 2: Key-value e.g. temp=28.5, light=65.0
    if "=" in line or ":" in line:
        try:
            parts = line.replace(";", ",").split(",")
            readings = {}
            for part in parts:
                sep = "=" if "=" in part else ":"
                k, v = part.split(sep, 1)
                k_clean = k.strip().lower()
                val = float(v.strip())
                if any(w in k_clean for w in ["temp", "temperature", "sensor_1", "s1"]) or k_clean == "t":
                    readings["temperature"] = val
                elif any(w in k_clean for w in ["light", "ldr", "lux", "sensor_2", "s2"]) or k_clean == "l":
                    readings["light"] = val
            if "temperature" in readings and "light" in readings:
                return readings
        except Exception:
            pass

    # Case 3: Comma separated "28.5, 65.0"
    if "," in line:
        try:
            parts = line.split(",")
            return {
                "temperature": float(parts[0].strip()),
                "light": float(parts[1].strip())
            }
        except Exception:
            pass

    return None


def run_bridge(port=None, baud=DEFAULT_BAUD, server_url=SERVER_URL):
    print("=" * 65)
    print("  Arduino Hardware Bridge (Temperature & Light/LDR Controller) ")
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
        print("You can test without hardware using: python run_simulator.py\n")
        return

    print(f"[Bridge Config] Target Serial Port: {port} @ {baud} baud")
    print(f"[Bridge Config] Target Server URL:  {server_url}\n")
    print(f"Connecting to Arduino on {port}...")

    try:
        ser = serial.Serial(port, baud, timeout=2)
        time.sleep(2)
        print(f" Connected to Arduino on {port}!")
        print("Streaming multi-modal telemetry to AI server... Press Ctrl+C to stop.\n")
    except Exception as e:
        print(f"[ERROR] Could not open serial port {port}: {e}")
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
                    print(f"[Arduino Debug] {line}")
                    continue

                temp = readings["temperature"]
                light = readings["light"]

                payload = {
                    "timestamp": int(time.time() * 1000),
                    "temperature": temp,
                    "light": light,
                    "source": f"arduino:{port}"
                }

                try:
                    res = requests.post(server_url, json=payload, timeout=2)
                    if res.status_code == 200:
                        res_data = res.json()
                        ctrl = res_data.get("control_signal", {})
                        insights = res_data.get("ai_insights", {})

                        led_1 = ctrl.get("led_1", 0)  # Thermal Load
                        led_2 = ctrl.get("led_2", 0)  # Smart Lighting
                        mode = ctrl.get("mode", "UNKNOWN")
                        pred_t = insights.get("predicted_temp", temp)
                        recon_err = insights.get("recon_loss", 0.0)
                        is_anom = insights.get("is_anomaly", False)

                        print(f"[TELEMETRY] Temp: {temp}C | Light: {light}%  ==>  [XGB Pred: {pred_t}C | Loss: {recon_err} | Anomaly: {is_anom}]")
                        print(f"[ACTUATE]   LED 1 (Thermal Load): {led_1} | LED 2 (Smart Light): {led_2} | Policy: {mode}\n")

                        # Send actuation command to Arduino
                        cmd = f'{{"led_1":{led_1},"led_2":{led_2}}}\n'
                        ser.write(cmd.encode('utf-8'))
                    else:
                        print(f"[WARN] Server HTTP {res.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"[WARN] AI server unreachable: {e}")

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[Bridge] Stopping hardware bridge. Turning off LEDs...")
        try:
            ser.write(b'{"led_1":0,"led_2":0}\n')
            time.sleep(0.5)
            ser.close()
        except Exception:
            pass
        print("[Bridge] Closed cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arduino Multi-Modal Bridge")
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g. COM3)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Baud rate (default 9600)")
    parser.add_argument("--url", type=str, default=SERVER_URL, help="Server URL")
    args = parser.parse_args()

    run_bridge(port=args.port, baud=args.baud, server_url=args.url)
