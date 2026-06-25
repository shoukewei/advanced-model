"""
plot_distributions.py — reads the NDJSON log written by generate_log.py
                         and produces two figures:
                           figures/ch13_latency_distribution.png
                           figures/ch13_prediction_distribution.png

Run from the project root after generate_log.py has completed:

    python plot_distributions.py
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, FIGURES

import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")

# ── Load log ───────────────────────────────────────────────────────────────
LOG_PATH = DATA_PROCESSED / "api_request_log.jsonl"
assert LOG_PATH.exists(), (
    f"{LOG_PATH.name} not found. Run generate_log.py first."
)

log_records = [json.loads(l) for l in LOG_PATH.read_text().strip().split("\n")]
log_df = pd.DataFrame(log_records)
log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])
print(f"Log loaded : {len(log_df)} records")

# ── Figure 1: Latency distribution ────────────────────────────────────────
latencies = log_df["latency_ms"].values
print(f"\nLatency (ms)"
      f"  p50={np.percentile(latencies,50):.1f}"
      f"  p95={np.percentile(latencies,95):.1f}"
      f"  p99={np.percentile(latencies,99):.1f}"
      f"  mean={latencies.mean():.1f}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(latencies, bins=30, color="steelblue", edgecolor="white")
for pct, ls in [(50, "--"), (95, "-."), (99, ":")]:
    v = np.percentile(latencies, pct)
    ax.axvline(v, linestyle=ls, color="crimson", lw=1.5, label=f"p{pct}={v:.1f}ms")
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Count")
ax.set_title("Inference Latency Distribution — 200 Requests")
ax.legend()
plt.tight_layout()
out1 = FIGURES / "ch13_latency_distribution.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved : {out1}")

# ── Figure 2: Prediction distribution ─────────────────────────────────────
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RANDOM_SEED

data  = load_breast_cancer()
X_all = pd.DataFrame(data.data, columns=data.feature_names)
y_all = pd.Series(data.target, name="malignant")
_, _, _, y_test = train_test_split(
    X_all, y_all, test_size=0.20,
    stratify=y_all, random_state=RANDOM_SEED
)

window = 20
rolling_pos = [
    log_df["prediction"].iloc[max(0, i - window):i].mean()
    for i in range(1, len(log_df) + 1)
]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(log_df["probability"], bins=25, color="steelblue", edgecolor="white")
axes[0].set_xlabel("Predicted P(malignant)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Predicted Probabilities")

axes[1].plot(rolling_pos, color="#E57373", lw=1.5)
axes[1].axvline(100, color="crimson", lw=1.5, linestyle="--", label="Drift at req 100")
axes[1].axhline(y_test.mean(), color="steelblue", lw=1, linestyle=":", label="True positive rate")
axes[1].set_xlabel("Request index")
axes[1].set_ylabel("Rolling fraction malignant")
axes[1].set_title(f"Rolling Predicted Class Balance (window={window})")
axes[1].legend(fontsize=8)

plt.suptitle("Prediction Distributions — 200 Requests")
plt.tight_layout()
out2 = FIGURES / "ch13_prediction_distribution.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved : {out2}")
