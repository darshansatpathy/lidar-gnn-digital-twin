"""Segmentation metrics."""

from __future__ import annotations

import numpy as np

from lidartwin.scenes import CLASSES


def confusion(truth: np.ndarray, pred: np.ndarray, n: int = len(CLASSES)) -> np.ndarray:
    return np.bincount(truth * n + pred, minlength=n * n).reshape(n, n)


def iou(conf: np.ndarray) -> np.ndarray:
    tp = np.diag(conf).astype(float)
    denom = conf.sum(axis=0) + conf.sum(axis=1) - tp
    return np.asarray(np.divide(tp, denom, out=np.full_like(tp, np.nan), where=denom > 0), dtype=float)


def summary(conf: np.ndarray) -> dict[str, float]:
    per = iou(conf)
    out = {"miou": float(np.nanmean(per)), "accuracy": float(np.trace(conf) / conf.sum())}
    out |= {f"iou_{c}": float(v) for c, v in zip(CLASSES, per, strict=True)}
    return out
