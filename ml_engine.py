"""
Real-Time AI/ML Engine for Dual-Sensor IoT Predictive Control and Anomaly Detection.
Maintains rolling window of sensor readings, computes dynamic lag features,
runs model inferences, and dispatches proactive control signals for actuators/LEDs.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from collections import deque

class MLEngine:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.predictor = None
        self.anomaly_detector = None
        self.metadata = {}
        self.control_mode = "AI_PREDICTIVE"  # Default: "AI_PREDICTIVE", can toggle to "CLASSICAL"
        
        # Sliding buffer for incoming telemetry (holds up to 20 samples)
        self.history = deque(maxlen=20)
        
        # State tracking
        self.tripped = False
        self.load_1 = 0
        self.load_2 = 0
        
        self._load_models()

    def _load_models(self):
        pred_path = os.path.join(self.models_dir, "model_predictor.pkl")
        iso_path = os.path.join(self.models_dir, "model_anomaly.pkl")
        meta_path = os.path.join(self.models_dir, "metadata.json")
        
        if os.path.exists(pred_path) and os.path.exists(iso_path):
            self.predictor = joblib.load(pred_path)
            self.anomaly_detector = joblib.load(iso_path)
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    self.metadata = json.load(f)
            print("[ML Engine] Pre-trained models loaded successfully.")
        else:
            print("[ML Engine] Models not found. Run 'train_models.py' first.")

    def set_control_mode(self, mode: str):
        """Toggle between 'AI_PREDICTIVE' and 'CLASSICAL'."""
        if mode in ["AI_PREDICTIVE", "CLASSICAL"]:
            self.control_mode = mode
            print(f"[ML Engine] Control mode switched to: {self.control_mode}")
            return True
        return False

    def process_telemetry(self, s1_temp: float, s1_hum: float, s2_temp: float, s2_hum: float):
        """
        Processes new dual-sensor telemetry reading and generates control signals.
        """
        # Store current reading in rolling history
        sample = {
            "s1_temp": float(s1_temp),
            "s1_hum": float(s1_hum),
            "s2_temp": float(s2_temp),
            "s2_hum": float(s2_hum),
            "load_1": self.load_1
        }
        self.history.append(sample)
        
        # Default fallback values if not enough history
        predicted_temp = round(s1_temp, 2)
        is_anomaly = False
        anomaly_score = 0.0
        anomaly_reason = "Normal"
        
        # Check rule-based sanity checks first (e.g. cross-sensor divergence)
        sensor_diff = abs(s1_temp - s2_temp)
        if sensor_diff > 5.5:
            is_anomaly = True
            anomaly_reason = f"Cross-Sensor Divergence ({sensor_diff:.1f}°C diff)"
        elif s1_temp > 45.0 or s2_temp > 45.0:
            is_anomaly = True
            anomaly_reason = "Extreme Temperature Spike"
            
        # Run ML models if available and enough history exists
        if len(self.history) >= 4 and self.predictor is not None and self.anomaly_detector is not None:
            features = self._extract_live_features()
            
            try:
                # 1. Predictive Regressor: Forecast t+5
                preds = self.predictor.predict(features)
                predicted_temp = round(float(preds[0]), 2)
                
                # 2. Isolation Forest Outlier Detection
                iso_pred = self.anomaly_detector.predict(features)[0]
                iso_score = float(self.anomaly_detector.decision_function(features)[0])
                anomaly_score = round(iso_score, 3)
                
                if iso_pred == -1 and not is_anomaly:
                    is_anomaly = True
                    anomaly_reason = "Multivariate Anomaly (Isolation Forest)"
            except Exception as e:
                print(f"[ML Engine Error] Inference failed: {e}")

        # Compute Control Signals based on selected mode
        if self.control_mode == "AI_PREDICTIVE":
            control_signals = self._compute_ai_control(s1_temp, s2_temp, predicted_temp, is_anomaly)
        else:
            control_signals = self._compute_classical_control(s1_temp, s2_temp)

        # Update internal load state
        self.load_1 = control_signals["led_1"]
        self.load_2 = control_signals["led_2"]
        self.tripped = control_signals["tripped"]

        return {
            "predicted_temp": predicted_temp,
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "anomaly_reason": anomaly_reason,
            "control_mode": self.control_mode,
            "led_1": self.load_1,             # Actuator 1: Primary Load (e.g. Heating or Relay)
            "led_2": self.load_2,             # Actuator 2: Safety Warning / Trip Alarm
            "load_state": "ON" if self.load_1 == 1 else "OFF",
            "tripped": self.tripped,
            "recommendation": control_signals["recommendation"]
        }

    def _extract_live_features(self):
        """Constructs single-row DataFrame matching the exact training feature space."""
        h = list(self.history)
        cur = h[-1]
        lag1 = h[-2]
        lag2 = h[-3]
        
        s1_rolling = np.mean([x["s1_temp"] for x in h[-4:]])
        s2_rolling = np.mean([x["s2_temp"] for x in h[-4:]])
        
        row = {
            "sensor_1_temp": cur["s1_temp"],
            "s1_temp_lag1": lag1["s1_temp"],
            "s1_temp_lag2": lag2["s1_temp"],
            "s1_delta": cur["s1_temp"] - lag1["s1_temp"],
            "s1_rolling_mean": s1_rolling,
            "sensor_2_temp": cur["s2_temp"],
            "s2_temp_lag1": lag1["s2_temp"],
            "s2_temp_lag2": lag2["s2_temp"],
            "s2_delta": cur["s2_temp"] - lag1["s2_temp"],
            "s2_rolling_mean": s2_rolling,
            "sensor_diff": abs(cur["s1_temp"] - cur["s2_temp"]),
            "sensor_1_hum": cur["s1_hum"],
            "sensor_2_hum": cur["s2_hum"],
            "load_1_state": lag1["load_1"]
        }
        return pd.DataFrame([row])

    def _compute_ai_control(self, s1_temp: float, s2_temp: float, predicted_temp: float, is_anomaly: bool):
        """
        Intelligent Proactive Control Policy:
        - Anticipates thermal breaches at t+5.
        - Sheds load BEFORE threshold breach (at 30.2°C forecast), avoiding harsh trips.
        """
        max_temp = max(s1_temp, s2_temp)
        
        # 1. Anomaly response: Safety alert
        if is_anomaly:
            return {
                "led_1": 0,
                "led_2": 1,
                "tripped": False,
                "recommendation": "ALERT: Anomaly detected, load suppressed for hardware safety."
            }
            
        # 2. Emergency hardware trip threshold (32.0°C)
        if max_temp >= 32.0:
            self.tripped = True
            return {
                "led_1": 0,
                "led_2": 1,
                "tripped": True,
                "recommendation": "CRITICAL: Hard threshold breach (≥32.0°C). System Tripped!"
            }
            
        # 3. AI Early Proactive Action: Forecast indicates imminent overheat
        if predicted_temp >= 30.4 or max_temp >= 30.2:
            return {
                "led_1": 0,
                "led_2": 0,
                "tripped": False,
                "recommendation": f"AI PROACTIVE ACTION: Forecast ({predicted_temp}°C) indicates heating trend. Load shed early to avoid trip."
            }
            
        # 4. Normal proactive operation
        if self.tripped and max_temp < 28.5:
            self.tripped = False
            
        led_1 = 1 if (not self.tripped and max_temp <= 29.5) else 0
        return {
            "led_1": led_1,
            "led_2": 0,
            "tripped": self.tripped,
            "recommendation": "OPTIMAL: Operating within AI energy-efficient bounds."
        }

    def _compute_classical_control(self, s1_temp: float, s2_temp: float):
        """
        Classical Reactive Threshold Policy:
        - Completely reactive: waits until temperature physically hits or exceeds 32.0°C.
        """
        max_temp = max(s1_temp, s2_temp)
        
        if max_temp >= 32.0:
            self.tripped = True
            return {
                "led_1": 0,
                "led_2": 1,
                "tripped": True,
                "recommendation": "CLASSICAL RULE: Overheat limit reached (≥32°C). Relay OFF, Trip LED ON."
            }
            
        if self.tripped and max_temp < 29.0:
            self.tripped = False
            return {
                "led_1": 1,
                "led_2": 0,
                "tripped": False,
                "recommendation": "CLASSICAL RULE: Recovered below 29°C. Relay reset ON."
            }
            
        if not self.tripped:
            led_1 = 1 if max_temp >= 28.0 else 0
            return {
                "led_1": led_1,
                "led_2": 0,
                "tripped": False,
                "recommendation": f"CLASSICAL RULE: Static threshold check. Load: {'ON' if led_1 else 'OFF'}."
            }
            
        return {
            "led_1": 0,
            "led_2": 1,
            "tripped": True,
            "recommendation": "CLASSICAL RULE: Waiting to cool down below 29°C."
        }
