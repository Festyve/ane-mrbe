# jspace-binding

Testing role–filler binding in the J-space global workspace.

Primary causal test: a byte-identical push along an in-house-fitted role axis
(`r_entity`), run in both signs; readout in log-odds; the lexical identity
swap survives only as the intervention-strength control. See
`docs/ARCHITECTURE.md` for the layout, the binding-score math, and the
GPU-day runbook.

```bash
pip install -e '.[dev]'
pytest

# GPU-free end-to-end validation against known ground truth:
python scripts/fit_directions.py --config configs/ci.yaml --dry-run
python scripts/calibrate.py      --config configs/ci.yaml --dry-run
python scripts/run_primary.py    --config configs/ci.yaml --dry-run --dummy-mode binding
python scripts/run_primary.py    --config configs/ci.yaml --dry-run --dummy-mode bag

# Secondary analyses (proposal §4):
python scripts/run_rq1.py        --config configs/ci.yaml --dry-run   # linear-probe selectivity
python scripts/run_rq2.py        --config configs/ci.yaml --dry-run   # ablation deltas

# Real-backend smoke test (tiny model + synthetic identity lens; needs '.[model]'):
python scripts/smoke_test.py
```
