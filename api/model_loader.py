import joblib
import os
import numpy as np
from tensorflow.keras.models import load_model as keras_load

MODEL_DIR = "models"

class ModelRegistry:
    """Holds all loaded models in memory. Loaded once at app startup."""

    def __init__(self):
        self.label_encoders = None
        self.random_forest = None
        self.prophet_models = {}
        self.lstm_models = {}
        self.lstm_scalers = {}
        self.loaded = False

    def load_all(self):
        print("Loading models into memory...")

        # Label encoders (used to convert strings -> numbers)
        enc_path = f"{MODEL_DIR}/label_encoders.pkl"
        if os.path.exists(enc_path):
            self.label_encoders = joblib.load(enc_path)
            print("  ✓ Label encoders loaded")

        # Random Forest baseline
        rf_path = f"{MODEL_DIR}/random_forest.pkl"
        if os.path.exists(rf_path):
            self.random_forest = joblib.load(rf_path)
            print("  ✓ Random Forest loaded")

        # Prophet models — one per route, filename pattern: prophet_<route>.pkl
        for fname in os.listdir(MODEL_DIR):
            if fname.startswith("prophet_") and fname.endswith(".pkl"):
                route_name = fname.replace("prophet_", "").replace(".pkl", "")
                self.prophet_models[route_name] = joblib.load(f"{MODEL_DIR}/{fname}")
                print(f"  ✓ Prophet model loaded: {route_name}")

        # LSTM models — filename pattern: lstm_<route>.keras
        for fname in os.listdir(MODEL_DIR):
            if fname.startswith("lstm_") and fname.endswith(".keras"):
                route_name = fname.replace("lstm_", "").replace(".keras", "")
                self.lstm_models[route_name] = keras_load(f"{MODEL_DIR}/{fname}")
                print(f"  ✓ LSTM model loaded: {route_name}")
            if fname.startswith("lstm_scaler_") and fname.endswith(".pkl"):
                route_name = fname.replace("lstm_scaler_", "").replace(".pkl", "")
                self.lstm_scalers[route_name] = joblib.load(f"{MODEL_DIR}/{fname}")

        self.loaded = True
        print(f"\nModels ready: RF={self.random_forest is not None}, "
              f"Prophet routes={len(self.prophet_models)}, "
              f"LSTM routes={len(self.lstm_models)}")

# Single shared instance used across the whole app
registry = ModelRegistry()
