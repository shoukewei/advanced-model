import json
import sys
from pathlib import Path
import pandas as pd

# Resolve project root (app/ → root) and import config
_APP_DIR = Path(__file__).resolve().parent
_ROOT    = _APP_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DATA_PROCESSED

LOG_PATH = DATA_PROCESSED / "api_request_log.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_request(request_id: str, features: dict, response: dict, latency_ms: float):
    """Append a request/response record as NDJSON."""
    record = {
        "request_id":    request_id,
        "timestamp":     pd.Timestamp.now().isoformat(),
        "features":      features,
        "prediction":    response["prediction"],
        "probability":   response["probability"],
        "latency_ms":    round(latency_ms, 3),
        "model_version": response.get("model_version"),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")