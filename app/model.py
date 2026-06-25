import joblib
from pathlib import Path
import pandas as pd
from .schemas import CancerFeatures

# Global model state
_model_state = {}

def load_model(model_path: Path):
    """Load LightGBM bundle from Chapter 12."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    bundle = joblib.load(model_path)
    
    _model_state["model"] = bundle["model"]
    _model_state["scaler"] = bundle["scaler"]
    _model_state["feature_names"] = bundle["feature_names"]
    _model_state["model_name"] = "lightgbm_bundle"
    _model_state["model_version"] = "1.0.0"
    _model_state["trained_at"] = bundle.get("trained_at")
    
    print(f"✅ Model loaded: {_model_state['model_name']} v{_model_state['model_version']}")
    return _model_state


def get_model_state():
    """Return current model state."""
    if "model" not in _model_state:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model_state


def run_inference(features: CancerFeatures) -> tuple[int, float]:
    """Scale features and run prediction."""
    state = get_model_state()
    
    # Convert Pydantic model to dict and map to original feature names
    feat_dict = features.model_dump()
    row = {k.replace("_", " "): v for k, v in feat_dict.items()}
    X_df = pd.DataFrame([row])[state["feature_names"]]
    
    X_sc = state["scaler"].transform(X_df)
    prob = float(state["model"].predict_proba(X_sc)[0, 1])
    pred = 1 if prob >= 0.5 else 0
    
    return pred, prob