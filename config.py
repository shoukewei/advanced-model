# config.py
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES        = ROOT / "figures"
MODELS         = ROOT / "models"
REPORTS        = ROOT / "reports"

# Create directories if they don't exist
for _dir in [DATA_RAW, DATA_PROCESSED, FIGURES, MODELS, REPORTS]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Modelling constants ────────────────────────────────────────────────
RANDOM_SEED   = 42
TEST_SIZE     = 0.2
CV_FOLDS      = 5
TARGET_COL    = "sales"

# ── Hyperparameter search space ────────────────────────────────────────
RIDGE_ALPHAS  = [0.01, 0.1, 1.0, 10.0, 100.0]