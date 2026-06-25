# ── app/schemas.py ─────────────────────────────────────────────────────────
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class CancerFeatures(BaseModel):
    """Input schema for the breast cancer classifier.

    All 30 features are required; values must be non-negative.
    Field names match sklearn's load_breast_cancer().feature_names
    with spaces replaced by underscores.
    """
    mean_radius:             float = Field(..., ge=0, description="Mean radius of cell nuclei")
    mean_texture:            float = Field(..., ge=0)
    mean_perimeter:          float = Field(..., ge=0)
    mean_area:               float = Field(..., ge=0)
    mean_smoothness:         float = Field(..., ge=0)
    mean_compactness:        float = Field(..., ge=0)
    mean_concavity:          float = Field(..., ge=0)
    mean_concave_points:     float = Field(..., ge=0)
    mean_symmetry:           float = Field(..., ge=0)
    mean_fractal_dimension:  float = Field(..., ge=0)
    radius_error:            float = Field(..., ge=0)
    texture_error:           float = Field(..., ge=0)
    perimeter_error:         float = Field(..., ge=0)
    area_error:              float = Field(..., ge=0)
    smoothness_error:        float = Field(..., ge=0)
    compactness_error:       float = Field(..., ge=0)
    concavity_error:         float = Field(..., ge=0)
    concave_points_error:    float = Field(..., ge=0)
    symmetry_error:          float = Field(..., ge=0)
    fractal_dimension_error: float = Field(..., ge=0)
    worst_radius:            float = Field(..., ge=0)
    worst_texture:           float = Field(..., ge=0)
    worst_perimeter:         float = Field(..., ge=0)
    worst_area:              float = Field(..., ge=0)
    worst_smoothness:        float = Field(..., ge=0)
    worst_compactness:       float = Field(..., ge=0)
    worst_concavity:         float = Field(..., ge=0)
    worst_concave_points:    float = Field(..., ge=0)
    worst_symmetry:          float = Field(..., ge=0)
    worst_fractal_dimension: float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    """Structured prediction response."""
    prediction:    int   = Field(..., description="0=benign, 1=malignant")
    probability:   float = Field(..., ge=0.0, le=1.0, description="P(malignant)")
    model_name:    str
    model_version: str
    latency_ms:    float = Field(..., description="Inference latency in ms")
    request_id:    str

class HealthResponse(BaseModel):
    status:        str
    model_name:    str
    model_version: str
    uptime_s:      float