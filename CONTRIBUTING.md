# Contributing

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
make check      # ruff, mypy, tests (includes a short training run)
```

## Changing the simulator, features or model

Every change that can move the numbers reruns the evaluation and commits the report:

```bash
ltw evaluate --report docs/evaluation-report.md
```

The gate fails below mIoU 0.85 on held-out streets, or when the network does not beat the gradient-boosting baseline. If you retrain, commit `models/dgcnn.pt` with its updated `models/dgcnn.json` (digest, data, settings) and say in the pull request what changed and why.

## Changing the twin

Asset extraction and change detection are tested against ground-truth labels in `tests/test_twin.py`, so a regression there is separate from the network's errors. Keep matching radii and the height threshold in `twin.py` and explain any change in an ADR.

## Commits

Small, focused commits with messages in the imperative. Update `CHANGELOG.md` for anything a user would notice.
