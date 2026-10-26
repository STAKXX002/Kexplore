"""
Model Training and Evaluation Pipeline for Dual-Sensor IoT Predictive Control.
Trains:
1. Predictive Forecaster (RandomForestRegressor) predicting sensor temp at t+5.
2. Multi-sensor Anomaly Detector (IsolationForest) identifying sensor drift and anomalies.
Saves models to the 'models/' directory.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score, classification_report

def extract_features(df):
    """
    Extracts lag features, temporal gradients, and rolling statistics for time-series prediction.
    """
    data = df.copy()
    
    # Lag features (t-1, t-2, t-3)
    data["s1_temp_lag1"] = data["sensor_1_temp"].shift(1)
    data["s1_temp_lag2"] = data["sensor_1_temp"].shift(2)
    data["s1_delta"] = data["sensor_1_temp"] - data["s1_temp_lag1"]
    data["s1_rolling_mean"] = data["sensor_1_temp"].rolling(window=4).mean()
    
    data["s2_temp_lag1"] = data["sensor_2_temp"].shift(1)
    data["s2_temp_lag2"] = data["sensor_2_temp"].shift(2)
    data["s2_delta"] = data["sensor_2_temp"] - data["s2_temp_lag1"]
    data["s2_rolling_mean"] = data["sensor_2_temp"].rolling(window=4).mean()
    
    # Cross-sensor discrepancy feature
    data["sensor_diff"] = (data["sensor_1_temp"] - data["sensor_2_temp"]).abs()
    
    data = data.dropna().reset_index(drop=True)
    return data

def train_and_export():
    dataset_path = "historical_training_data.csv"
    from generate_dataset import generate_telemetry_dataset
    generate_telemetry_dataset(num_samples=4000, output_file=dataset_path)
        
    print(f"\nLoading dataset from '{dataset_path}'...")
    raw_df = pd.read_csv(dataset_path)
    df = extract_features(raw_df)
    
    # Define feature columns
    feature_cols = [
        "sensor_1_temp", "s1_temp_lag1", "s1_temp_lag2", "s1_delta", "s1_rolling_mean",
        "sensor_2_temp", "s2_temp_lag1", "s2_temp_lag2", "s2_delta", "s2_rolling_mean",
        "sensor_diff", "sensor_1_hum", "sensor_2_hum", "load_1_state"
    ]
    
    X = df[feature_cols]
    y_reg = df["target_s1_temp_next5"]
    y_anomaly = df["is_anomaly"]
    
    # Split train/test (chronological 80/20 split)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    
    # -------------------------------------------------------------
    # 1. Train Predictive Temperature Regressor
    # -------------------------------------------------------------
    print("\n--- Training Predictive Regressor (Random Forest) ---")
    regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Predictive Model Test R² Score: {r2:.4f}")
    print(f"Predictive Model Test MAE:      {mae:.4f} °C")
    
    # -------------------------------------------------------------
    # 2. Train Anomaly Detection Model (Isolation Forest)
    # -------------------------------------------------------------
    print("\n--- Training Dual-Sensor Anomaly Detector (Isolation Forest) ---")
    # Train on normal baseline samples from train split
    normal_mask = (y_anomaly.iloc[:split_idx] == 0)
    X_train_normal = X_train[normal_mask]
    
    anomaly_detector = IsolationForest(
        n_estimators=120,
        contamination=0.035,
        random_state=42
    )
    anomaly_detector.fit(X_train_normal)
    
    # Evaluate anomaly detector on test split
    test_iso_preds = anomaly_detector.predict(X_test)
    test_pred_binary = np.where(test_iso_preds == -1, 1, 0)
    y_test_anomaly = y_anomaly.iloc[split_idx:]
    
    print("\nAnomaly Detection Evaluation on Test Set:")
    print(classification_report(y_test_anomaly, test_pred_binary, labels=[0, 1], target_names=["Normal", "Anomaly"], zero_division=0))
    
    # -------------------------------------------------------------
    # 3. Export Models and Metadata
    # -------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    reg_path = os.path.join("models", "model_predictor.pkl")
    iso_path = os.path.join("models", "model_anomaly.pkl")
    meta_path = os.path.join("models", "metadata.json")
    
    joblib.dump(regressor, reg_path)
    joblib.dump(anomaly_detector, iso_path)
    
    metadata = {
        "features": feature_cols,
        "metrics": {
            "r2_score": round(float(r2), 4),
            "mae": round(float(mae), 4)
        },
        "thresholds": {
            "temp_trip_threshold": 32.0,
            "ai_predictive_cooling_threshold": 30.2,
            "sensor_discrepancy_limit": 5.0
        }
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nModels successfully saved to 'models/' directory.")
    print(f" - Predictor: {reg_path}")
    print(f" - Anomaly Detector: {iso_path}")
    print(f" - Metadata: {meta_path}")

if __name__ == "__main__":
    train_and_export()
