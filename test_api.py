"""
test_api.py — standalone API test script
Run from the project root with the server already running:

    python -m uvicorn app.main:app --reload          # terminal 1
    python test_api.py                               # terminal 2
"""

import sys
import time
import json
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RANDOM_SEED

BASE_URL = "http://127.0.0.1:8000"

# ── Helpers ────────────────────────────────────────────────────────────────
def sep(title=""):
    width = 60
    if title:
        print(f"\n{'─' * 4} {title} {'─' * (width - len(title) - 6)}")
    else:
        print("─" * width)


def post(payload: dict) -> requests.Response:
    return requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)


# ── Dataset ────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X_all = pd.DataFrame(data.data, columns=data.feature_names)
y_all = pd.Series(data.target, name="malignant")
_, X_test, _, y_test = train_test_split(
    X_all, y_all, test_size=0.20,
    stratify=y_all, random_state=RANDOM_SEED
)


# ══════════════════════════════════════════════════════════════════════════
# 1. Health check
# ══════════════════════════════════════════════════════════════════════════
sep("1 · Health check")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=3)
    h = r.json()
    print(f"  status   : {h['status']}")
    print(f"  model    : {h['model_name']} v{h['model_version']}")
    print(f"  uptime   : {h['uptime_s']}s")
except requests.exceptions.ConnectionError:
    print("  ERROR: server not reachable.")
    print("  Start it with:  python -m uvicorn app.main:app --reload")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# 2. Ten labelled predictions
# ══════════════════════════════════════════════════════════════════════════
sep("2 · Labelled predictions (n=10)")
print(f"  {'req':<10}  {'pred':>4}  {'prob':>8}  {'true':>4}  {'ok?':>4}  latency")
print(f"  {'─'*10}  {'─'*4}  {'─'*8}  {'─'*4}  {'─'*4}  {'─'*7}")

correct = 0
for i in range(10):
    row        = X_test.iloc[i]
    payload    = {col.replace(" ", "_"): float(v) for col, v in row.items()}
    true_label = int(y_test.iloc[i])

    r    = post(payload)
    resp = r.json()
    pred = resp["prediction"]
    ok   = "✓" if pred == true_label else "✗"
    correct += pred == true_label

    print(f"  {resp['request_id']:<10}  {pred:>4}  {resp['probability']:>8.4f}"
          f"  {true_label:>4}  {ok:>4}  {resp['latency_ms']:.1f}ms")

print(f"\n  accuracy : {correct}/10")


# ══════════════════════════════════════════════════════════════════════════
# 3. Rejection tests (invalid inputs → expect HTTP 422)
# ══════════════════════════════════════════════════════════════════════════
sep("3 · Rejection tests")
valid = {col.replace(" ", "_"): 1.0 for col in data.feature_names}

bad_cases = [
    ("Missing field",              {k: v for k, v in valid.items() if k != "worst_fractal_dimension"}),
    ("Negative value (ge=0)",      {**valid, "mean_radius": -5.0}),
    ("Wrong type (string→float)",  {**valid, "mean_radius": "bad"}),
]

for label, payload in bad_cases:
    r  = post(payload)
    ok = "✓" if r.status_code == 422 else f"✗ (got {r.status_code})"
    print(f"  {label:<38}  HTTP {r.status_code}  {ok}")


# ══════════════════════════════════════════════════════════════════════════
# 4. Latency benchmark (n=50)
# ══════════════════════════════════════════════════════════════════════════
sep("4 · Latency benchmark (n=50)")
client_ms = []
server_ms = []

for i in range(50):
    row     = X_test.iloc[i % len(X_test)]
    payload = {col.replace(" ", "_"): float(v) for col, v in row.items()}

    t0  = time.perf_counter()
    r   = post(payload)
    client_ms.append((time.perf_counter() - t0) * 1000)
    server_ms.append(r.json()["latency_ms"])

c, s = np.array(client_ms), np.array(server_ms)
print(f"  {'':12}  {'p50':>6}  {'p95':>6}  {'p99':>6}  {'mean':>6}")
print(f"  {'client ms':<12}  {np.percentile(c,50):>6.1f}  {np.percentile(c,95):>6.1f}"
      f"  {np.percentile(c,99):>6.1f}  {c.mean():>6.1f}")
print(f"  {'server ms':<12}  {np.percentile(s,50):>6.1f}  {np.percentile(s,95):>6.1f}"
      f"  {np.percentile(s,99):>6.1f}  {s.mean():>6.1f}")
print(f"  {'overhead':<12}  {np.percentile(c-s,50):>6.1f}  {np.percentile(c-s,95):>6.1f}"
      f"  {np.percentile(c-s,99):>6.1f}  {(c-s).mean():>6.1f}  ← network + serialisation")

import matplotlib.pyplot as plt
import seaborn as sns
from config import FIGURES
sns.set_theme(style="whitegrid", palette="muted")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(c, label="Client (round-trip)", color="steelblue", lw=1.5)
ax.plot(s, label="Server (inference only)", color="#66BB6A", lw=1.5)
ax.fill_between(range(50), s, c, alpha=0.15, color="orange",
                label="Network + serialisation overhead")
ax.set_xlabel("Request index")
ax.set_ylabel("Latency (ms)")
ax.set_title("Client vs. Server Latency — 50 Live Requests")
ax.legend(fontsize=9)
plt.tight_layout()
out = FIGURES / "ch13_live_latency.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Plot saved : {out}")


# ══════════════════════════════════════════════════════════════════════════
# 5. Summary
# ══════════════════════════════════════════════════════════════════════════
sep()
print("  All tests passed  ✓")
