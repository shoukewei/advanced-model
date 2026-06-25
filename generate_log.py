"""
generate_log.py — sends 200 requests to the live API and writes the
                   NDJSON log used by Chapter 14 monitoring analysis.

Requests 0–99   : in-distribution (drawn from the test set)
Requests 100–199: covariate shift injected
                  (mean_radius ×1.4, mean_area ×1.4, worst_radius ×1.3)

Run from the project root with the server already running:

    python -m uvicorn app.main:app --reload    # terminal 1
    python generate_log.py                     # terminal 2
"""

import sys
import json
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RANDOM_SEED, DATA_PROCESSED

BASE_URL = "http://127.0.0.1:8000"
LOG_PATH = DATA_PROCESSED / "api_request_log.jsonl"

# ── Check server ───────────────────────────────────────────────────────────
try:
    r = requests.get(f"{BASE_URL}/health", timeout=3)
    h = r.json()
    print(f"Server   : {BASE_URL}  ({h['model_name']} v{h['model_version']})")
except requests.exceptions.ConnectionError:
    print("ERROR: server not reachable.")
    print("Start it with:  python -m uvicorn app.main:app --reload")
    sys.exit(1)

# ── Dataset ────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X_all = pd.DataFrame(data.data, columns=data.feature_names)
y_all = pd.Series(data.target, name="malignant")
_, X_test, _, y_test = train_test_split(
    X_all, y_all, test_size=0.20,
    stratify=y_all, random_state=RANDOM_SEED
)

# ── Send 200 requests ──────────────────────────────────────────────────────
LOG_PATH.unlink(missing_ok=True)
all_true_labels = []
rng = np.random.default_rng(RANDOM_SEED)

print(f"Sending 200 requests (drift injected at request 100)...")
print(f"{'req':<5}  {'id':<10}  {'pred':>4}  {'prob':>8}  {'true':>4}  {'ok?':>4}  window")

for i in range(200):
    idx = i % len(X_test)
    row = X_test.iloc[idx].copy()

    if i >= 100:
        row["mean radius"]  *= 1.4
        row["mean area"]    *= 1.4
        row["worst radius"] *= 1.3

    true_label = int(y_test.iloc[idx])
    payload    = {col.replace(" ", "_"): float(v) for col, v in row.items()}

    r    = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
    resp = r.json()
    pred = resp["prediction"]
    ok   = "✓" if pred == true_label else "✗"
    window = "in-dist" if i < 100 else "drifted"

    # Print every 10th request as progress
    if i % 10 == 0 or i == 199:
        print(f"{i:>3}    {resp['request_id']:<10}  {pred:>4}  "
              f"{resp['probability']:>8.4f}  {true_label:>4}  {ok:>4}  {window}")

    all_true_labels.append(true_label)

# ── Attach true labels and rewrite log ────────────────────────────────────
log_records = [json.loads(l) for l in LOG_PATH.read_text().strip().split("\n")]
for rec, lbl in zip(log_records, all_true_labels):
    rec["true_label"] = lbl
with open(LOG_PATH, "w") as f:
    for rec in log_records:
        f.write(json.dumps(rec) + "\n")

log_df = pd.DataFrame(log_records)
print(f"\nLog written  : {LOG_PATH}")
print(f"Records      : {len(log_df)}  (0–99 in-dist, 100–199 covariate-shifted)")
print(f"Columns      : {list(log_df.columns)}")
