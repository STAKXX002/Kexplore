"""
Central Configuration for Multi-Modal AI IoT Telemetry & Actuation System.
Provides path resolution, networking endpoints, and physical threshold defaults.
"""

import os

# Root project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Asset and storage directories
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "src", "server", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "src", "server", "static")

# File paths
CSV_FILE = os.path.join(DATA_DIR, "sensor_dataset.csv")
TRAINING_DATA_FILE = os.path.join(DATA_DIR, "historical_training_data.csv")

# Model weight file paths
MODEL_XGB_PATH = os.path.join(MODELS_DIR, "model_xgboost.json")
MODEL_AE_PATH = os.path.join(MODELS_DIR, "model_autoencoder.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

# Server networking configurations
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5000
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}/telemetry"
TARGET_URL = f"http://127.0.0.1:{SERVER_PORT}/api/set-target"
DEMO_URL = f"http://127.0.0.1:{SERVER_PORT}/api/demo/inject-state"

# Arduino serial defaults
DEFAULT_BAUD = 9600

# Physical operating defaults
DEFAULT_TARGET_TEMP = 27.0
DEFAULT_TARGET_LIGHT = 50.0
DEFAULT_TEMP_TRIP = 32.0
DEFAULT_AI_COOLING = 30.2
DEFAULT_LIGHT_MIN = 35.0

# Ensure data and models directory exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
