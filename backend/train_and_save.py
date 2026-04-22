"""
Compatibility wrapper for Colab/local training.

This script now delegates to the main Bengaluru hybrid trainer so older run
instructions do not accidentally train on the legacy Delhi/monthdata path.
"""

import os

from train_hybrid_models import resolve_data_path, train_hybrid_model


MODEL_DIR = "backend/models"


def train_models():
    data_path = resolve_data_path("backend/data/karnataka_realdata.csv")
    os.makedirs(MODEL_DIR, exist_ok=True)
    return train_hybrid_model(data_path=data_path, model_dir=MODEL_DIR, epochs=60)


if __name__ == "__main__":
    train_models()
