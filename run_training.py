"""
Runs the multi-modal training pipeline for XGBoost and PyTorch Deep Autoencoder.
Usage:
    python run_training.py
"""

from src.ml.train import train_and_export

if __name__ == '__main__':
    train_and_export()
