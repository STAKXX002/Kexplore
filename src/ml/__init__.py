"""
Machine Learning & Predictive Control Subsystem.
Includes:
- MLEngine: Real-time multi-modal inference and counterfactual MPC.
- MultimodalAutoencoder: PyTorch deep neural autoencoder for cross-sensor anomaly detection.
- train_and_export: Pipeline to retrain XGBoost and PyTorch autoencoder models.
- generate_telemetry_dataset: Physics-based training data generator.
"""

from src.ml.engine import MLEngine, MultimodalAutoencoder
from src.ml.train import train_and_export
from src.ml.dataset_generator import generate_telemetry_dataset

__all__ = [
    "MLEngine",
    "MultimodalAutoencoder",
    "train_and_export",
    "generate_telemetry_dataset",
]
