# How to test API in Chapter 13

Place `test_api.py` in the project root alongside `config.py`, then:

## Terminal 1 — keep running
```bash
python -m uvicorn app.main:app --reload
```
# Terminal 1 — keep running throughout
python -m uvicorn app.main:app --reload

# Terminal 2 — run in sequence
`python test_api.py`     # health + 10 labelled + rejection + 50-req benchmark
`python generate_log.py` # 200-request stream with drift at req 100 → writes log
`python plot_distributions.py` # reads log → saves latency + prediction figures

Expected output:
```
──── 1 · Health check ───────────────────────────────────
  status   : ok
  model    : lightgbm_bundle v1.0.0
  uptime   : 4.2s

──── 2 · Labelled predictions (n=10) ────────────────────
  req           pred      prob  true   ok?  latency
  ──────────    ────  ────────  ────  ────  ───────
  6c76abeb         0    0.0000     0     ✓  1.2ms
  ...

──── 3 · Rejection tests ─────────────────────────────────
  Missing field                           HTTP 422  ✓
  Negative value (ge=0)                   HTTP 422  ✓
  Wrong type (string→float)               HTTP 422  ✓

──── 4 · Latency benchmark (n=50) ───────────────────────
                 p50     p95     p99    mean
  client ms      8.1    12.4    15.0     8.6
  server ms      1.2     2.1     2.8     1.4
  overhead       6.9    10.3    12.2     7.2  ← network + serialisation
────────────────────────────────────────────────────────
  All tests passed  ✓
```

# Open Browser
```
http://127.0.0.1:8000/docs
```
It already add a root redirect to `/docs`:

`http://127.0.0.1:8000/` redirect to it in `main.py`.

```
http://127.0.0.1:8000/health
```


`main1.py` without direct.