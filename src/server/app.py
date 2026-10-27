"""
Central Flask Controller & Telemetry API Server.
Coordinates multi-modal telemetry ingestion, real-time ML inference,
SCADA analytics logging, dynamic REST reconfigurations, and Web Dashboard.
"""

from flask import Flask, request, jsonify, render_template, send_file
import csv
import os
import time
import logging
from datetime import datetime
from collections import deque

try:
    from src.config import (
        CSV_FILE,
        MODELS_DIR,
        TEMPLATES_DIR,
        STATIC_DIR,
        SERVER_HOST,
        SERVER_PORT,
    )
    from src.ml.engine import MLEngine
    from src.server.analytics import SessionAnalytics, DEMO_MODES
except ImportError:
    from config import (
        CSV_FILE,
        MODELS_DIR,
        TEMPLATES_DIR,
        STATIC_DIR,
        SERVER_HOST,
        SERVER_PORT,
    )
    from ml.engine import MLEngine
    from server.analytics import SessionAnalytics, DEMO_MODES


# Suppress ONLY repetitive HTTP request access lines ('GET ...', 'POST ...')
# but keep server startup messages like 'Running on http://...'
class NoHttpRequestFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not ('GET ' in msg or 'POST ' in msg)


logging.getLogger('werkzeug').addFilter(NoHttpRequestFilter())

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR,
    static_url_path='/static'
)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Initialize Multi-Modal AI Engine (XGBoost + PyTorch Deep Autoencoder)
ml_engine = MLEngine(models_dir=MODELS_DIR)

# Session metrics, SCADA audit trail, link heartbeat and demo-injection state
analytics = SessionAnalytics(stability_band=0.35)

# In-memory history for real-time dashboard visualization
live_history = deque(maxlen=40)
current_latest = None

CSV_HEADERS = [
    "timestamp", "temperature", "light", "predicted_temp", "recon_loss",
    "is_anomaly", "anomaly_reason", "led_1", "led_2",
    "load_state", "smart_light_state", "tripped", "control_mode"
]


def _ensure_csv_schema():
    """
    Creates the dataset file with the current header. If an existing file was
    written by an older version (different header), it is renamed to a
    timestamped legacy backup instead of mixing two column layouts.
    """
    os.makedirs(os.path.dirname(os.path.abspath(CSV_FILE)), exist_ok=True)
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="r", newline="") as f:
            first_row = next(csv.reader(f), None)
        if first_row == CSV_HEADERS:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = os.path.dirname(os.path.abspath(CSV_FILE))
        legacy_name = os.path.join(dir_name, f"sensor_dataset_legacy_{stamp}.csv")
        os.replace(CSV_FILE, legacy_name)
        print(f"[Server] Existing CSV had an outdated header. Moved to {legacy_name}")
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)


_ensure_csv_schema()


@app.route('/')
@app.route('/dashboard')
def dashboard():
    """Serves the live interactive multi-modal AI control dashboard."""
    return render_template('dashboard.html')


@app.route('/telemetry', methods=['POST'])
def receive_telemetry():
    """
    Receives multi-modal telemetry from:
    1. Physical Arduino via hardware bridge
    2. Synthetic simulation via simulator
    """
    global current_latest
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400

    timestamp = data.get("timestamp", int(time.time() * 1000))

    # Extract Temperature and Light with flexible alias support
    temperature = data.get("temperature", data.get("temp", data.get("sensor_1", 25.0)))
    light = data.get("light", data.get("ldr", data.get("sensor_2", 50.0)))

    temperature = float(temperature)
    light = float(light)

    # Process through XGBoost + PyTorch Deep Autoencoder (timed for latency metric)
    t_start = time.perf_counter()
    ai_result = ml_engine.process_telemetry(
        temperature=temperature,
        light=light
    )
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    analytics.record_sample(
        temperature=temperature,
        light=light,
        result=ai_result,
        latency_ms=latency_ms,
        timestamp_ms=timestamp,
        source=data.get("source")
    )

    pred_temp = ai_result["predicted_temp"]
    recon_loss = ai_result["recon_loss"]
    is_anomaly = ai_result["is_anomaly"]
    anomaly_reason = ai_result["anomaly_reason"]
    led_1 = ai_result["led_1"]
    led_2 = ai_result["led_2"]
    load_state = ai_result["load_state"]
    smart_light_state = ai_result["smart_light_state"]
    tripped = ai_result["tripped"]
    control_mode = ai_result["control_mode"]

    # Log to CSV
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, temperature, light, pred_temp, recon_loss,
            is_anomaly, anomaly_reason, led_1, led_2,
            load_state, smart_light_state, tripped, control_mode
        ])

    time_str = datetime.fromtimestamp(timestamp / 1000.0).strftime('%H:%M:%S')
    latest_record = {
        "timestamp": timestamp,
        "time_str": time_str,
        "temperature": temperature,
        "light": light,
        "predicted_temp": pred_temp,
        "recon_loss": recon_loss,
        "is_anomaly": is_anomaly,
        "anomaly_reason": anomaly_reason,
        "target_temp": ai_result["target_temp"],
        "target_light": ai_result["target_light"],
        "temp_error": ai_result["temp_error"],
        "target_achieved": ai_result["target_achieved"],
        "led_1": led_1,
        "led_2": led_2,
        "load_state": load_state,
        "smart_light_state": smart_light_state,
        "tripped": tripped,
        "control_mode": control_mode,
        "recommendation": ai_result["recommendation"]
    }
    current_latest = latest_record
    live_history.append(latest_record)

    # Terminal Telemetry Output (ASCII safe for Windows console)
    status_label = "[ANOMALY]" if is_anomaly else ("[TRIPPED]" if tripped else ("[TARGET REACHED]" if ai_result["target_achieved"] else "[OPTIMAL]"))
    print(f"[{control_mode}] Temp: {temperature:.1f}C (Target: {ai_result['target_temp']}C) | Light: {light:.1f}% | XGB Pred: {pred_temp:.1f}C | LEDs: [{led_1}, {led_2}] | {status_label}")

    return jsonify({
        "status": "success",
        "control_signal": {
            "led_1": led_1,
            "led_2": led_2,
            "load_state": load_state,
            "smart_light_state": smart_light_state,
            "tripped": tripped,
            "mode": control_mode
        },
        "ai_insights": {
            "predicted_temp": pred_temp,
            "recon_loss": recon_loss,
            "is_anomaly": is_anomaly,
            "anomaly_reason": anomaly_reason,
            "target_temp": ai_result["target_temp"],
            "target_light": ai_result["target_light"],
            "temp_error": ai_result["temp_error"],
            "target_achieved": ai_result["target_achieved"],
            "recommendation": ai_result["recommendation"],
            "latency_ms": round(latency_ms, 2)
        },
        # Active presentation scenario; simulator follows this value
        "demo_mode": analytics.demo_mode
    }), 200


@app.route('/api/live-status', methods=['GET'])
def get_live_status():
    """Provides current multi-modal status, active thresholds, history and session analytics for dashboard."""
    return jsonify({
        "latest": current_latest,
        "history": list(live_history),
        "thresholds": ml_engine.get_thresholds(),
        "analytics": analytics.snapshot()
    })


@app.route('/api/session-summary', methods=['GET'])
def get_session_summary():
    """Returns cumulative statistics for the current lab run."""
    return jsonify({"status": "success", "summary": analytics.summary()}), 200


@app.route('/api/events', methods=['GET'])
def get_events():
    """Returns the SCADA audit trail (newest first)."""
    return jsonify({"status": "success", "events": analytics.snapshot()["events"]}), 200


@app.route('/api/export-csv', methods=['GET'])
def export_csv():
    """Downloads the full logged telemetry dataset as a CSV attachment."""
    path = os.path.abspath(CSV_FILE)
    if not os.path.exists(path):
        return jsonify({"status": "error", "message": "No dataset recorded yet"}), 404
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(path, mimetype="text/csv", as_attachment=True,
                     download_name=f"telemetry_{stamp}.csv")


@app.route('/api/demo/current-mode', methods=['GET'])
def get_demo_mode():
    """Returns the active presentation demo scenario."""
    return jsonify({"status": "success", "mode": analytics.demo_mode, "available": DEMO_MODES}), 200


@app.route('/api/demo/inject-state', methods=['POST'])
def inject_demo_state():
    """
    Switches the presentation demo scenario (simulator only).
    Accepts JSON: {"mode": "NORMAL" | "DARK_HEAT_SPIKE" | "SOLAR_SURGE" | "LDR_FAULT"}
    """
    data = request.json or {}
    mode = str(data.get("mode", "")).upper()
    if not analytics.set_demo_mode(mode):
        return jsonify({"status": "error", "message": f"Invalid mode. Use one of {DEMO_MODES}"}), 400
    return jsonify({"status": "success", "mode": mode}), 200


@app.route('/api/thresholds', methods=['GET'])
def get_thresholds():
    """Returns currently active trip and actuation thresholds."""
    return jsonify({
        "status": "success",
        "thresholds": ml_engine.get_thresholds()
    }), 200


@app.route('/api/target', methods=['GET'])
def get_target():
    """Returns currently active goal target setpoints."""
    return jsonify({
        "status": "success",
        "target": ml_engine.get_target()
    }), 200


@app.route('/api/set-target', methods=['POST'])
def set_target():
    """
    API endpoint to update the goal target that AI regulates toward.
    Accepts JSON:
    {
        "target_temp": 26.5,    # Target temperature setpoint in °C
        "target_light": 50.0    # Target illumination setpoint in %
    }
    """
    data = request.json or {}
    target_temp = data.get("target_temp", data.get("temp_target"))
    target_light = data.get("target_light", data.get("light_target"))
    updated = ml_engine.update_target(target_temp=target_temp, target_light=target_light)
    analytics.log_event("OPERATOR", f"Setpoint changed: target {updated['target_temp']}C / light {updated['target_light']}%", "info")
    return jsonify({
        "status": "success",
        "message": "Goal target setpoints updated successfully",
        "target": updated
    }), 200


@app.route('/api/set-thresholds', methods=['POST'])
def set_thresholds():
    """
    API endpoint to dynamically change emergency trip, actuation thresholds, and targets.
    """
    data = request.json or {}
    temp_trip = data.get("temp_trip", data.get("temp_trip_threshold"))
    ai_cooling = data.get("ai_cooling", data.get("ai_cooling_threshold"))
    light_min = data.get("light_min", data.get("light_min_threshold"))
    target_temp = data.get("target_temp", data.get("temp_target"))
    target_light = data.get("target_light", data.get("light_target"))

    updated = ml_engine.update_thresholds(
        temp_trip=temp_trip,
        ai_cooling=ai_cooling,
        light_min=light_min,
        target_temp=target_temp,
        target_light=target_light
    )
    analytics.log_event(
        "OPERATOR",
        f"Config updated: target {updated['target_temp']}C, trip {updated['temp_trip_threshold']}C, "
        f"AI preempt {updated['ai_cooling_threshold']}C, light {updated['target_light']}%",
        "info"
    )
    return jsonify({
        "status": "success",
        "message": "Configuration updated successfully",
        "thresholds": updated
    }), 200


@app.route('/api/toggle-mode', methods=['POST'])
def toggle_mode():
    data = request.json or {}
    new_mode = data.get("mode", "AI_PREDICTIVE")
    success = ml_engine.set_control_mode(new_mode)
    if success:
        label = "AI Model Predictive Control" if new_mode == "AI_PREDICTIVE" else "Classical Bang-Bang"
        analytics.log_event("OPERATOR", f"Control mode switched to {label}", "warning" if new_mode == "CLASSICAL" else "ai")
        return jsonify({"status": "success", "mode": new_mode}), 200
    return jsonify({"status": "error", "message": "Invalid mode"}), 400


def start_server(host=SERVER_HOST, port=SERVER_PORT):
    print("=" * 65)
    print("Multi-Modal AI IoT Central Controller (XGBoost + PyTorch Autoencoder)")
    print(f"  -> Web Dashboard:  http://127.0.0.1:{port}/dashboard")
    print(f"  -> Telemetry API:  http://127.0.0.1:{port}/telemetry")
    print("=" * 65)
    print(f"[Server] Listening on http://{host}:{port} (Press Ctrl+C to stop)\n")
    try:
        app.run(host=host, port=port, debug=False)
    except KeyboardInterrupt:
        print("\n[Server] Stopped by user (Ctrl+C).")
    except Exception as e:
        print(f"\n[FATAL SERVER ERROR]: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    start_server()
