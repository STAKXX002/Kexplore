# Kexplore: Multi-Modal AI IoT Telemetry & Actuation System

An industrial-grade IoT system combining physical sensor acquisition (Temperature & Light/LDR) with machine learning: **XGBoost Regressor Model Predictive Control (MPC)** and a **PyTorch Deep Autoencoder Neural Network** for cross-sensor anomaly detection, supervisory safeguards, and real-time SCADA telemetry visualization.

---

## Architecture & Directory Structure

```text
Kexplore/
├── src/                               # Core Application Source Code
│   ├── config.py                      # Central configuration (paths, endpoints, defaults)
│   ├── ml/                            # Machine Learning Subsystem
│   │   ├── engine.py                  # MLEngine: Real-time inference & Counterfactual MPC
│   │   ├── train.py                   # Model training pipeline (XGBoost + PyTorch Autoencoder)
│   │   └── dataset_generator.py       # Thermodynamic synthetic dataset generator
│   ├── server/                        # Central Telemetry Server & SCADA Web Dashboard
│   │   ├── app.py                     # Flask REST API server & routing
│   │   ├── analytics.py               # SCADA session metrics, event audit trail, duty-cycle
│   │   ├── templates/                 # Web interface markup
│   │   │   └── dashboard.html         # Responsive dark-theme dashboard HTML
│   │   └── static/                    # Modular dashboard assets
│   │       ├── css/dashboard.css      # Custom styling & animations
│   │       └── js/dashboard.js        # Chart.js, polling & UI logic
│   ├── hardware/                      # Physical Embedded & Microcontroller Layer
│   │   ├── bridge.py                  # Serial bridge forwarding Arduino packets to central server
│   │   └── arduino_controller.ino     # Arduino C++ firmware sketch (A0/A1 reading, Pin 8/9 PWM)
│   └── simulation/                    # Synthetic Presentation Testing
│       └── simulator.py               # Environmental simulator with interactive scenario keys
│
├── models/                            # Serialized Model Artifacts
│   ├── model_xgboost.json             # Trained XGBoost trajectory forecaster
│   ├── model_autoencoder.pth          # PyTorch Deep Autoencoder weights
│   ├── scaler.pkl                     # Scikit-learn feature standardizer
│   └── metadata.json                  # Model thresholds and feature definitions
│
├── data/                              # Data Storage
│   ├── historical_training_data.csv   # Training dataset
│   └── sensor_dataset.csv             # Runtime telemetry logs
│
├── tests/                             # Automated Test Suite
│   └── test_system.py                 # Comprehensive unit & integration tests
│
├── run_server.py                      # Launcher: Central Web Controller
├── run_simulator.py                   # Launcher: Telemetry Simulator
├── run_bridge.py                      # Launcher: Arduino Serial Bridge
├── run_training.py                    # Launcher: Retrain AI Models
│
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## Quick Start Guide

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Start the Central Web Controller
```bash
python run_server.py
```
Open **`http://127.0.0.1:5000/dashboard`** in your browser.

### 3. Stream Telemetry Data

#### Option A: Software Simulation (No Hardware Needed)
```bash
python run_simulator.py
```
* **Interactive Presentation Controls in Simulator Terminal:**
  - Press `1`: Trigger thermal surge in darkness (PyTorch Autoencoder anomaly trip)
  - Press `2`: Trigger intense solar heat wave (XGBoost preemptive cooling)
  - Press `3`: Trigger LDR sensor disconnect fault
  - Press `4`: Step target setpoint to `29.5°C` (heating pursuit)
  - Press `5`: Step target setpoint to `26.0°C` (cooling pursuit)
  - Press `0`: Reset to normal dynamics (`27.0°C`)

#### Option B: Physical Arduino Hardware
Upload `src/hardware/arduino_controller.ino` to your Arduino board and connect via USB, then run:
```bash
python run_bridge.py
```

### 4. Run Automated Test Suite
```bash
python -m unittest discover tests
```

---

## Core Capabilities

1. **AI Model Predictive Control (MPC)**:
   - Uses **XGBoost** to evaluate counterfactual trajectories ($u=0$ coasting vs $u=1$ heating).
   - Anticipates thermal inertia and shuts off heating load early to eliminate overshoot.
2. **PyTorch Deep Autoencoder Anomaly Safeguard**:
   - Compresses 10 multi-modal features into a 3D latent bottleneck.
   - Computes real-time reconstruction loss ($MSE$) to detect abnormal cross-sensor states (e.g., heat surge in darkness).
3. **SCADA Event Audit Trail**:
   - Auto-scrolling, color-coded audit stream logging AI decisions, setpoint locks, and emergency overrides.
4. **Energy Efficiency & Stability Metrics**:
   - Calculates live Load Duty Saved (%) compared to classical thermostats, setpoint stability (% within $\pm 0.35^\circ\text{C}$), and AI inference latency ($< 10\,\text{ms}$).
5. **Interactive In-Browser Presentation Bar**:
   - 1-click scenario buttons on the dashboard to demo faults and recoveries directly on a projector.
