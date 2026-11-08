from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "dgcnn.pt"

needs_model = pytest.mark.skipif(not MODEL.exists(), reason="trained model not present")
