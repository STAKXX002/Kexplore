"""
Real-Time Multi-Modal AI Engine (XGBoost + PyTorch Deep Autoencoder).
Heterogeneous Sensor Fusion: Temperature (°C) + Light/LDR (%).
Dual Actuation: LED 1 (Thermal Load) + LED 2 (Smart Illumination/Alert).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from collections import deque
import xgboost as xgb
import torch
import torch.nn as nn

try:
    from src.config import (
        MODELS_DIR,
        DEFAULT_TARGET_TEMP,
        DEFAULT_TARGET_LIGHT,
        DEFAULT_TEMP_TRIP,
        DEFAULT_AI_COOLING,
        DEFAULT_LIGHT_MIN,
    )
except ImportError:
    from config import (
        MODELS_DIR,
        DEFAULT_TARGET_TEMP,
        DEFAULT_TARGET_LIGHT,
        DEFAULT_TEMP_TRIP,
        DEFAULT_AI_COOLING,
        DEFAULT_LIGHT_MIN,
    )


# Re-use exact PyTorch Autoencoder definition
class MultimodalAutoencoder(nn.Module):
    def __init__(self, input_dim=10, latent_dim=3):
        super(MultimodalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class MLEngine:
    def __init__(self, models_dir=None):
        self.models_dir = models_dir or MODELS_DIR
        self.xgb_model = None
        self.autoencoder = None
        self.scaler = None
        self.metadata = {}
        self.recon_threshold = 0.25
        self.control_mode = "AI_PREDICTIVE"

        # Configurable Dynamic Thresholds & Target Setpoints
        self.target_temp = DEFAULT_TARGET_TEMP       # Desired target temperature setpoint (°C)
        self.target_light = DEFAULT_TARGET_LIGHT     # Desired target illumination setpoint (%)
        self.temp_trip_threshold = DEFAULT_TEMP_TRIP # Emergency safety ceiling (°C)
        self.ai_cooling_threshold = DEFAULT_AI_COOLING # Preemptive cooling trigger (°C)
        self.light_min_threshold = DEFAULT_LIGHT_MIN  # Daylight threshold (%)

        # History buffer for lag and rolling features
        self.history = deque(maxlen=25)

        # Actuator states
        self.led_1 = 0      # Thermal Relay / Cooling (1: ON/Heating, 0: OFF/Cooling)
        self.led_2 = 0      # Smart Illumination (1: ON, 0: OFF)
        self.tripped = False

        self._load_models()

    def _load_models(self):
        xgb_path = os.path.join(self.models_dir, "model_xgboost.json")
        ae_path = os.path.join(self.models_dir, "model_autoencoder.pth")
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        meta_path = os.path.join(self.models_dir, "metadata.json")

        if os.path.exists(xgb_path) and os.path.exists(ae_path) and os.path.exists(scaler_path):
            # 1. Load XGBoost
            self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model(xgb_path)

            # 2. Load Feature Scaler
            self.scaler = joblib.load(scaler_path)

            # 3. Load Metadata
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    self.metadata = json.load(f)
                    self.recon_threshold = self.metadata.get("metrics", {}).get("autoencoder_threshold", 0.25)
                    thresholds = self.metadata.get("thresholds", {})
                    self.temp_trip_threshold = thresholds.get("temp_trip_threshold", self.temp_trip_threshold)
                    self.ai_cooling_threshold = thresholds.get("ai_predictive_cooling_threshold", self.ai_cooling_threshold)
                    self.light_min_threshold = thresholds.get("smart_light_min_threshold", self.light_min_threshold)

            # 4. Load PyTorch Autoencoder
            input_dim = self.metadata.get("input_dim", 10)
            latent_dim = self.metadata.get("latent_dim", 3)
            self.autoencoder = MultimodalAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
            self.autoencoder.load_state_dict(torch.load(ae_path, map_location=torch.device('cpu')))
            self.autoencoder.eval()

            print("[ML Engine] XGBoost and PyTorch Deep Autoencoder loaded successfully.")
        else:
            print(f"[ML Engine] Model weights not found in '{self.models_dir}'. Run training first.")

    def update_thresholds(self, temp_trip=None, ai_cooling=None, light_min=None, target_temp=None, target_light=None):
        """Dynamically reconfigures trip, actuation thresholds, and target setpoints via API request."""
        if temp_trip is not None:
            self.temp_trip_threshold = round(float(temp_trip), 2)
        if ai_cooling is not None:
            self.ai_cooling_threshold = round(float(ai_cooling), 2)
        if light_min is not None:
            self.light_min_threshold = round(float(light_min), 2)
        if target_temp is not None:
            self.target_temp = round(float(target_temp), 2)
        if target_light is not None:
            self.target_light = round(float(target_light), 2)
        print(f"[ML Engine] Updated via API: Target={self.target_temp}C | Trip={self.temp_trip_threshold}C | Target Light={self.target_light}%")
        return self.get_thresholds()

    def update_target(self, target_temp=None, target_light=None):
        """Sets the active goal target that AI model predictive control regulates toward."""
        if target_temp is not None:
            self.target_temp = round(float(target_temp), 2)
        if target_light is not None:
            self.target_light = round(float(target_light), 2)
        print(f"[ML Engine] Target Setpoint Updated: Temp Target={self.target_temp}C | Light Target={self.target_light}%")
        return self.get_target()

    def get_target(self):
        """Returns active goal target setpoints."""
        return {
            "target_temp": self.target_temp,
            "target_light": self.target_light
        }

    def get_thresholds(self):
        """Returns currently active threshold configuration & target setpoints."""
        return {
            "target_temp": self.target_temp,
            "target_light": self.target_light,
            "temp_trip_threshold": self.temp_trip_threshold,
            "ai_cooling_threshold": self.ai_cooling_threshold,
            "light_min_threshold": self.light_min_threshold,
            "recon_threshold": self.recon_threshold
        }

    def set_control_mode(self, mode: str):
        if mode in ["AI_PREDICTIVE", "CLASSICAL"]:
            self.control_mode = mode
            print(f"[ML Engine] Switched mode to: {self.control_mode}")
            return True
        return False

    def process_telemetry(self, temperature: float, light: float):
        """
        Processes new multi-modal telemetry reading:
        - Temperature in °C
        - Light / LDR in % (0 - 100)
        """
        sample = {
            "temperature": float(temperature),
            "light": float(light),
            "led_1": self.led_1
        }
        self.history.append(sample)

        predicted_temp = round(temperature, 2)
        pred_if_off = predicted_temp
        pred_if_on = predicted_temp
        temp_delta = (self.history[-1]["temperature"] - self.history[-2]["temperature"]) if len(self.history) >= 2 else 0.0
        is_anomaly = False
        recon_loss = 0.0
        anomaly_reason = "Normal"

        # Heuristic cross-sensor sanity checks
        if light <= 10.0 and temperature >= 32.0:
            is_anomaly = True
            anomaly_reason = "Abnormal Thermal Runaway in Darkness (Fire/Fault Risk)"
        elif light <= 0.5 or light >= 99.5:
            if len(self.history) >= 4 and abs(self.history[-1]["temperature"] - self.history[-3]["temperature"]) < 0.5:
                is_anomaly = True
                anomaly_reason = "LDR Light Sensor Saturation / Disconnect Fault"

        # Multi-modal Model Inferences
        if len(self.history) >= 4 and self.xgb_model is not None and self.autoencoder is not None:
            features = self._extract_live_features()

            try:
                # 1. Counterfactual XGBoost MPC Forecasts (t+5 under u=0 vs u=1)
                cf_batch = pd.concat([features, features], ignore_index=True)
                cf_batch.loc[0, "led_1"] = 0
                cf_batch.loc[1, "led_1"] = 1
                preds = self.xgb_model.predict(cf_batch.values)
                pred_if_off = round(float(preds[0]), 2)
                pred_if_on = round(float(preds[1]), 2)
                predicted_temp = pred_if_on if self.led_1 == 1 else pred_if_off

                # 2. PyTorch Deep Autoencoder (Reconstruction Loss Anomaly Scoring)
                features_scaled = self.scaler.transform(features.values)
                tensor_in = torch.tensor(features_scaled, dtype=torch.float32)

                with torch.no_grad():
                    recons = self.autoencoder(tensor_in).numpy()
                    mse = float(np.mean(np.square(features_scaled - recons)))
                    recon_loss = round(mse, 5)

                if recon_loss > self.recon_threshold:
                    is_anomaly = True
                    if anomaly_reason == "Normal":
                        anomaly_reason = f"Neural Autoencoder Anomaly (Loss: {recon_loss} > {self.recon_threshold:.4f})"

            except Exception as e:
                print(f"[ML Engine Error] Inference error: {e}")

        # Compute Dual Actuator Control Policy
        if self.control_mode == "AI_PREDICTIVE":
            control_signals = self._compute_ai_control(
                temperature, light, pred_if_off, pred_if_on, temp_delta, is_anomaly, recon_loss
            )
        else:
            control_signals = self._compute_classical_control(temperature, light)

        self.led_1 = control_signals["led_1"]
        self.led_2 = control_signals["led_2"]
        self.tripped = control_signals["tripped"]
        if len(self.history) >= 4 and self.xgb_model is not None:
            predicted_temp = pred_if_on if self.led_1 == 1 else pred_if_off

        return {
            "predicted_temp": predicted_temp,
            "recon_loss": recon_loss,
            "is_anomaly": is_anomaly,
            "anomaly_reason": anomaly_reason,
            "control_mode": self.control_mode,
            "target_temp": self.target_temp,
            "target_light": self.target_light,
            "temp_error": round(temperature - self.target_temp, 2),
            "target_achieved": abs(temperature - self.target_temp) <= 0.35,
            "led_1": self.led_1,
            "led_2": self.led_2,
            "load_state": "ON" if self.led_1 == 1 else "OFF",
            "smart_light_state": "ON" if self.led_2 == 1 else "OFF",
            "tripped": self.tripped,
            "recommendation": control_signals["recommendation"]
        }

    def _extract_live_features(self):
        h = list(self.history)
        cur = h[-1]
        lag1 = h[-2]
        lag2 = h[-3]

        temp_rolling = np.mean([x["temperature"] for x in h[-4:]])
        solar_flux = round(cur["temperature"] * (cur["light"] / 100.0), 2)

        row = {
            "temperature": cur["temperature"],
            "temp_lag1": lag1["temperature"],
            "temp_lag2": lag2["temperature"],
            "temp_delta": cur["temperature"] - lag1["temperature"],
            "temp_rolling": temp_rolling,
            "light": cur["light"],
            "light_lag1": lag1["light"],
            "light_delta": cur["light"] - lag1["light"],
            "solar_thermal_flux": solar_flux,
            "led_1": cur["led_1"]
        }
        return pd.DataFrame([row])

    def _compute_ai_control(
        self,
        temperature: float,
        light: float,
        pred_if_off: float,
        pred_if_on: float,
        temp_delta: float,
        is_anomaly: bool,
        recon_loss: float,
    ):
        """
        Counterfactual AI Model Predictive Control (MPC) Policy:
        - Evaluates XGBoost 5-step-ahead trajectory under u=0 (coasting) vs u=1 (heating).
        - Shuts off thermal load early when residual thermal momentum coasts into target.
        - Regulates smart illumination toward self.target_light.
        - Emergency trip and anomaly safeguards act as supervisory overrides.
        """
        # Actuator 2: Smart Daylight Harvesting toward target illumination
        led_2 = 1 if (light < self.target_light or is_anomaly or self.tripped) else 0

        # 1. Anomaly safeguard override
        if is_anomaly:
            return {
                "led_1": 0,
                "led_2": 1,
                "tripped": False,
                "recommendation": "SAFETY ALERT: Deep Autoencoder detected anomalous pattern. Thermal load cut off."
            }

        # 2. Emergency trip ceiling override
        if temperature >= self.temp_trip_threshold:
            self.tripped = True
            return {
                "led_1": 0,
                "led_2": 1,
                "tripped": True,
                "recommendation": f"CRITICAL: Emergency trip threshold breach (≥{self.temp_trip_threshold}°C). System Tripped!"
            }

        # 3. Preemptive Overheat Protection (Safety boundary)
        if pred_if_off >= self.ai_cooling_threshold or temperature >= (self.ai_cooling_threshold - 0.2):
            return {
                "led_1": 0,
                "led_2": led_2,
                "tripped": False,
                "recommendation": f"XGBOOST PREEMPTIVE COOLING: Overheat avoidance ({pred_if_off}°C ≥ {self.ai_cooling_threshold}°C)."
            }

        if self.tripped and temperature < (self.temp_trip_threshold - 3.5):
            self.tripped = False

        # 4. Counterfactual MPC Target Point Regulation
        temp_error = round(temperature - self.target_temp, 2)
        is_at_target = abs(temp_error) <= 0.35

        if not self.tripped:
            if is_at_target:
                # Hold setpoint tightly: prefer u=0 unless coasting forecast drops below band
                if pred_if_off < (self.target_temp - 0.15) and pred_if_on <= (self.target_temp + 0.35):
                    led_1 = 1
                    rec_text = f"TARGET REGULATING: Applying micro-pulse to sustain {self.target_temp}°C."
                elif pred_if_off > (self.target_temp + 0.15) or temp_delta > 0.05:
                    led_1 = 0
                    rec_text = f"TARGET LOCKED ({self.target_temp}°C): AI coasting load to prevent overshoot."
                else:
                    led_1 = 0
                    rec_text = f"TARGET ACHIEVED ({temperature:.1f}°C): Operating precisely at {self.target_temp}°C setpoint."
            elif temperature < self.target_temp:
                # Below target: shut off early ONLY when approaching with positive momentum
                # and XGBoost predicts residual inertia will coast into target (or ON would overshoot)
                gap = self.target_temp - temperature
                can_coast = (
                    gap <= 0.85
                    and temp_delta > 0.0
                    and (pred_if_off >= (self.target_temp - 0.12) or pred_if_on > (self.target_temp + 0.32))
                )
                if can_coast:
                    led_1 = 0
                    rec_text = f"AI PREDICTIVE CONTROL: Forecast ({pred_if_off}°C) indicates thermal coasting to {self.target_temp}°C. Early shut-off."
                else:
                    led_1 = 1
                    rec_text = f"AI TARGET TRACKING: Driving temperature toward {self.target_temp}°C (Current: {temperature:.1f}°C)."
            else:
                # Above target: shed heat to descend toward setpoint
                led_1 = 0
                rec_text = f"AI REGULATING TO TARGET: Shedding heat to descend to {self.target_temp}°C (Current: {temperature:.1f}°C)."
        else:
            led_1 = 0
            rec_text = "TRIPPED: Cooling down below safety reset threshold."

        return {
            "led_1": led_1,
            "led_2": led_2,
            "tripped": self.tripped,
            "recommendation": rec_text
        }

    def _compute_classical_control(self, temperature: float, light: float):
        """Classical bang-bang control (no predictive compensation; causes overshoot)."""
        if temperature >= self.temp_trip_threshold:
            self.tripped = True
            return {
                "led_1": 0,
                "led_2": 1,
                "tripped": True,
                "recommendation": f"CLASSICAL RULE: Overheat limit (≥{self.temp_trip_threshold}°C). Tripped."
            }

        if self.tripped and temperature < (self.temp_trip_threshold - 3.0):
            self.tripped = False
            return {
                "led_1": 1,
                "led_2": 0,
                "tripped": False,
                "recommendation": f"CLASSICAL RULE: Reset below {self.temp_trip_threshold - 3.0}°C."
            }

        # Reactive thermostat: turns off only AFTER crossing target
        led_1 = 1 if (not self.tripped and temperature < self.target_temp) else 0
        led_2 = 1 if light < self.target_light else 0

        status_text = "Heater ON (Below target)" if led_1 == 1 else "Heater OFF (Above target)"
        return {
            "led_1": led_1,
            "led_2": led_2,
            "tripped": self.tripped,
            "recommendation": f"CLASSICAL BANG-BANG: {status_text}. No predictive compensation."
        }
