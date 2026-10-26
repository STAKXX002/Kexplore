from flask import Flask, request, jsonify, render_template
import csv
import os
import time
import logging
from datetime import datetime
from collections import deque
from ml_engine import MLEngine

# Suppress Werkzeug default HTTP request access logs (GET / POST lines)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)
CSV_FILE = "sensor_dataset.csv"

# Initialize AI/ML Engine
ml_engine = MLEngine(models_dir="models")

# In-memory history for real-time dashboard visualization (stores last 40 readings)
live_history = deque(maxlen=40)
current_latest = None

# Initialize CSV file with multi-sensor and AI telemetry headers
CSV_HEADERS = [
    "timestamp", "sensor_1_temp", "sensor_1_hum", "sensor_2_temp", "sensor_2_hum",
    "predicted_temp", "is_anomaly", "anomaly_reason", "led_1", "led_2",
    "load_state", "tripped", "control_mode"
]

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)

@app.route('/')
@app.route('/dashboard')
def dashboard():
    """Serves the live interactive AI control dashboard."""
    return render_template('dashboard.html')

@app.route('/telemetry', methods=['POST'])
def receive_telemetry():
    """
    Receives multi-sensor telemetry from either:
    1. Physical Arduino via arduino_bridge.py
    2. Synthetic simulation via syn_sender.py
    3. Legacy single-sensor devices
    """
    global current_latest
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400

    timestamp = data.get("timestamp", int(time.time() * 1000))
    
    # Support dual-sensor format or fallback gracefully
    # If explicit sensor_1 / sensor_2 keys are present:
    if "sensor_1" in data or "sensor_2" in data:
        s1_temp = float(data.get("sensor_1", 25.0))
        s2_temp = float(data.get("sensor_2", s1_temp))
        s1_hum = float(data.get("humidity_1", data.get("humidity", 50.0)))
        s2_hum = float(data.get("humidity_2", s1_hum))
    else:
        # Backward compatibility with single sensor payload
        s1_temp = float(data.get("temperature", 25.0))
        s2_temp = float(data.get("temperature_2", s1_temp))
        s1_hum = float(data.get("humidity", 50.0))
        s2_hum = float(data.get("humidity_2", s1_hum))

    # Process through Real-time AI / ML Engine
    ai_result = ml_engine.process_telemetry(
        s1_temp=s1_temp,
        s1_hum=s1_hum,
        s2_temp=s2_temp,
        s2_hum=s2_hum
    )

    pred_temp = ai_result["predicted_temp"]
    is_anomaly = ai_result["is_anomaly"]
    anomaly_reason = ai_result["anomaly_reason"]
    led_1 = ai_result["led_1"]
    led_2 = ai_result["led_2"]
    load_state = ai_result["load_state"]
    tripped = ai_result["tripped"]
    control_mode = ai_result["control_mode"]

    # Log to CSV
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, s1_temp, s1_hum, s2_temp, s2_hum,
            pred_temp, is_anomaly, anomaly_reason,
            led_1, led_2, load_state, tripped, control_mode
        ])

    # Store in memory for live web dashboard
    time_str = datetime.fromtimestamp(timestamp / 1000.0).strftime('%H:%M:%S')
    latest_record = {
        "timestamp": timestamp,
        "time_str": time_str,
        "sensor_1_temp": s1_temp,
        "sensor_1_hum": s1_hum,
        "sensor_2_temp": s2_temp,
        "sensor_2_hum": s2_hum,
        "predicted_temp": pred_temp,
        "is_anomaly": is_anomaly,
        "anomaly_reason": anomaly_reason,
        "led_1": led_1,
        "led_2": led_2,
        "load_state": load_state,
        "tripped": tripped,
        "control_mode": control_mode,
        "recommendation": ai_result["recommendation"]
    }
    current_latest = latest_record
    live_history.append(latest_record)

    # Console feedback
    print(f"[{control_mode}] S1: {s1_temp}°C | S2: {s2_temp}°C | Pred: {pred_temp}°C | LEDs: [{led_1}, {led_2}] | Anomaly: {is_anomaly}")

    # Return dynamic control signals back to the sender / Arduino bridge
    return jsonify({
        "status": "success",
        "control_signal": {
            "led_1": led_1,
            "led_2": led_2,
            "load_state": load_state,
            "tripped": tripped,
            "mode": control_mode
        },
        "ai_insights": {
            "predicted_temp": pred_temp,
            "is_anomaly": is_anomaly,
            "anomaly_score": ai_result["anomaly_score"],
            "anomaly_reason": anomaly_reason,
            "recommendation": ai_result["recommendation"]
        }
    }), 200

@app.route('/api/live-status', methods=['GET'])
def get_live_status():
    """Provides current telemetry snapshot and recent historical points for the web chart."""
    return jsonify({
        "latest": current_latest,
        "history": list(live_history)
    })

@app.route('/api/toggle-mode', methods=['POST'])
def toggle_mode():
    """API endpoint to switch between 'AI_PREDICTIVE' and 'CLASSICAL' mode."""
    data = request.json or {}
    new_mode = data.get("mode", "AI_PREDICTIVE")
    success = ml_engine.set_control_mode(new_mode)
    if success:
        return jsonify({"status": "success", "mode": new_mode}), 200
    return jsonify({"status": "error", "message": "Invalid mode. Choose 'AI_PREDICTIVE' or 'CLASSICAL'"}), 400

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Central Dual-Sensor AI Controller Server")
    print("Web Dashboard:  http://127.0.0.1:5000/dashboard")
    print("Telemetry API:  http://127.0.0.1:5000/telemetry")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
