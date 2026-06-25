import sys
from pathlib import Path

# Add the project root to Python path so we can import config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # Goes from app/ → root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now you can import config
from config import DATA_PROCESSED, FIGURES, RANDOM_SEED, MODELS

import uuid
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from .schemas import CancerFeatures, PredictionResponse, HealthResponse
from .model import load_model, run_inference, get_model_state
from .logger import log_request

# Configuration

MODEL_PATH = MODELS / "lightgbm_bundle.joblib"

SERVER_START = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup."""
    print("🚀 Starting Cancer Classifier API...")
    try:
        load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
    yield
    print("👋 Shutting down API...")

# Create FastAPI app
app = FastAPI(
    title="Breast Cancer Classifier API",
    description="Production-ready LightGBM model serving",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", include_in_schema=False)
async def root():
    """Redirect / to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Health check endpoint."""
    try:
        state = get_model_state()
        return HealthResponse(
            status="ok",
            model_name=state["model_name"],
            model_version=state["model_version"],
            uptime_s=round(time.time() - SERVER_START, 1),
        )
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(features: CancerFeatures):
    """Predict malignancy for a single observation."""
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()
    
    try:
        pred, prob = run_inference(features)
        latency_ms = (time.perf_counter() - t0) * 1000
        
        response = {
            "prediction": pred,
            "probability": round(prob, 6),
            "model_name": "lightgbm_bundle",
            "model_version": "1.0.0",
            "latency_ms": round(latency_ms, 3),
            "request_id": request_id
        }
        
        # Log the request
        log_request(request_id, features.model_dump(), response, latency_ms)
        
        return PredictionResponse(**response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))