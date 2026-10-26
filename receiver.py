from flask import Flask, request, jsonify
import csv
import os

app = Flask(__name__)
CSV_FILE = "sensor_dataset.csv"

# Initialize CSV file with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "temperature", "humidity", "load_state", "mode", "tripped"])

@app.route('/telemetry', methods=['POST'])
def receive_telemetry():
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400

    timestamp = data.get("timestamp")
    temp = data.get("temperature")
    hum = data.get("humidity")
    load = data.get("load_state")
    mode = data.get("mode")
    tripped = data.get("tripped")

    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, temp, hum, load, mode, tripped])

    print(f"Logged -> Temp: {temp}°C | Hum: {hum}% | Load: {load} | Mode: {mode}")
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
