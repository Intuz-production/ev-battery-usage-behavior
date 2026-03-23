"""
Model loading utilities
"""
from pathlib import Path
from tensorflow.keras.models import load_model

def load_models(soh_model_path: Path, rul_model_path: Path):
    """
    Load SOH and RUL models from disk

    Args:
        soh_model_path: Path to SOH model file
        rul_model_path: Path to RUL model file

    Returns:
        Tuple of (soh_model, rul_model)
    """
    try:
        soh_model = load_model(soh_model_path)
        rul_model = load_model(rul_model_path)
        return soh_model, rul_model
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}")