"""
Model Training and Evaluation Pipeline.
Trains:
1. XGBoost Regressor (GitHub dmlc/xgboost): Cross-Modal Solar-Thermal Predictive Forecaster
2. PyTorch Deep Autoencoder Neural Network: Multimodal Anomaly & Hazard Detector
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, classification_report

# -------------------------------------------------------------
# PyTorch Deep Multimodal Autoencoder Architecture
# -------------------------------------------------------------
class MultimodalAutoencoder(nn.Module):
    def __init__(self, input_dim=10, latent_dim=3):
        super(MultimodalAutoencoder, self).__init__()
        # Encoder: Compresses 10 multi-modal features down to a 3D latent bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, latent_dim)
        )
        # Decoder: Reconstructs the original multi-modal features from latent space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

def extract_features(df):
    """Extracts lag features, temporal gradients, and cross-modal solar-thermal flux."""
    data = df.copy()
    
    # Temperature lag & gradients
    data["temp_lag1"] = data["temperature"].shift(1)
    data["temp_lag2"] = data["temperature"].shift(2)
    data["temp_delta"] = data["temperature"] - data["temp_lag1"]
    data["temp_rolling"] = data["temperature"].rolling(window=4).mean()
    
    # Light lag & gradients
    data["light_lag1"] = data["light"].shift(1)
    data["light_delta"] = data["light"] - data["light_lag1"]
    data["light_rolling"] = data["light"].rolling(window=4).mean()
    
    # Cross-Modal Interaction: Solar-Thermal Flux (Light intensity * Temp)
    data["solar_thermal_flux"] = (data["temperature"] * (data["light"] / 100.0)).round(2)
    
    data = data.dropna().reset_index(drop=True)
    return data

def train_and_export():
    dataset_path = "historical_training_data.csv"
    from generate_dataset import generate_telemetry_dataset
    generate_telemetry_dataset(num_samples=4200, output_file=dataset_path)
    
    print(f"\nLoading multi-modal dataset from '{dataset_path}'...")
    raw_df = pd.read_csv(dataset_path)
    df = extract_features(raw_df)
    
    feature_cols = [
        "temperature", "temp_lag1", "temp_lag2", "temp_delta", "temp_rolling",
        "light", "light_lag1", "light_delta", "solar_thermal_flux", "led_1"
    ]
    
    X = df[feature_cols].values
    y_reg = df["target_temp_next5"].values
    y_anomaly = df["is_anomaly"].values
    
    # Chronological 80/20 train/test split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_reg[:split_idx], y_reg[split_idx:]
    y_test_anomaly = y_anomaly[split_idx:]
    
    # -------------------------------------------------------------
    # 1. Train XGBoost Regressor (GitHub dmlc/xgboost)
    # -------------------------------------------------------------
    print("\n--- [1] Training XGBoost Regressor (Predictive Forecasting) ---")
    xgb_model = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"XGBoost Test R² Score: {r2:.4f}")
    print(f"XGBoost Test MAE:      {mae:.4f} °C")
    
    # Feature Importances
    importances = xgb_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nXGBoost Feature Importances (Cross-Modal Attribution):")
    for idx in sorted_idx[:5]:
        print(f" - {feature_cols[idx]:<20}: {importances[idx]*100:.2f}%")
        
    # -------------------------------------------------------------
    # 2. Train PyTorch Deep Autoencoder Neural Network
    # -------------------------------------------------------------
    print("\n--- [2] Training PyTorch Deep Autoencoder (Multi-Modal Anomaly Detection) ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train only on normal baseline data
    normal_mask = (y_anomaly[:split_idx] == 0)
    X_train_normal = X_train_scaled[normal_mask]
    
    device = torch.device("cpu")
    input_dim = len(feature_cols)
    autoencoder = MultimodalAutoencoder(input_dim=input_dim, latent_dim=3).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005, weight_decay=1e-5)
    
    train_tensor = torch.tensor(X_train_normal, dtype=torch.float32)
    dataset = TensorDataset(train_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    autoencoder.train()
    for epoch in range(40):
        total_loss = 0.0
        for batch in dataloader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(inputs)
            
    # Calculate baseline normal reconstruction error threshold (mean + 3.0 * std)
    autoencoder.eval()
    with torch.no_grad():
        normal_recons = autoencoder(train_tensor).numpy()
        normal_mse = np.mean(np.square(X_train_normal - normal_recons), axis=1)
        recon_threshold = float(np.mean(normal_mse) + 3.0 * np.std(normal_mse))
        
    print(f"PyTorch Autoencoder Baseline MSE: {np.mean(normal_mse):.5f}")
    print(f"Calculated Anomaly Threshold (mean + 3*std): {recon_threshold:.5f}")
    
    # Evaluate Anomaly Detection on Test Split
    with torch.no_grad():
        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        test_recons = autoencoder(test_tensor).numpy()
        test_mse = np.mean(np.square(X_test_scaled - test_recons), axis=1)
        test_pred_anomaly = (test_mse > recon_threshold).astype(int)
        
    print("\nPyTorch Autoencoder Anomaly Classification Report:")
    print(classification_report(y_test_anomaly, test_pred_anomaly, labels=[0, 1], target_names=["Normal", "Anomaly"], zero_division=0))
    
    # -------------------------------------------------------------
    # 3. Export Models & Metadata
    # -------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    xgb_path = os.path.join("models", "model_xgboost.json")
    ae_path = os.path.join("models", "model_autoencoder.pth")
    scaler_path = os.path.join("models", "scaler.pkl")
    meta_path = os.path.join("models", "metadata.json")
    
    xgb_model.save_model(xgb_path)
    torch.save(autoencoder.state_dict(), ae_path)
    joblib.dump(scaler, scaler_path)
    
    metadata = {
        "features": feature_cols,
        "input_dim": input_dim,
        "latent_dim": 3,
        "metrics": {
            "xgboost_r2": round(float(r2), 4),
            "xgboost_mae": round(float(mae), 4),
            "autoencoder_threshold": round(float(recon_threshold), 5)
        },
        "thresholds": {
            "temp_trip_threshold": 32.0,
            "ai_predictive_cooling_threshold": 30.2,
            "smart_light_min_threshold": 35.0
        }
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nAll models successfully trained and exported:")
    print(f" - XGBoost Regressor:      {xgb_path}")
    print(f" - PyTorch Autoencoder:    {ae_path}")
    print(f" - Feature Scaler:         {scaler_path}")
    print(f" - System Metadata:        {meta_path}")

if __name__ == "__main__":
    train_and_export()
