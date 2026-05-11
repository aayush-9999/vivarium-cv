# pipeline/threshold/engine.py
"""
Converts a continuous fill percentage to a status string.

Thresholds are loaded once from CONFIG at import time.
pct_to_status() is the public API used by the measurer and pipeline.
"""

from __future__ import annotations

from core.config_loader import CONFIG


def _build_thresholds() -> dict[str, list[tuple[float, float, str]]]:
    """Build sorted (lo, hi, status) tuples for each container type."""
    raw = CONFIG["thresholds"]
    result: dict[str, list[tuple[float, float, str]]] = {}
    for container, mapping in raw.items():
        result[container] = [
            (lo, hi, status.upper())
            for status, (lo, hi) in mapping.items()
        ]
    return result


_THRESHOLDS = _build_thresholds()


def pct_to_status(pct: float, container: str = "water") -> str:
    """
    Map a continuous fill percentage to OK / LOW / CRITICAL.

    Parameters
    ----------
    pct : float
        Fill percentage in [0.0, 100.0].
    container : str
        "water" or "food" — selects the threshold set.

    Returns
    -------
    str : "OK" | "LOW" | "CRITICAL"
    """
    for lo, hi, status in _THRESHOLDS.get(container, _THRESHOLDS["water"]):
        if lo <= pct < hi:
            return status
    return "CRITICAL"