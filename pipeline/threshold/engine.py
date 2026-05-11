# pipeline/threshold/engine.py
from core.config_loader import CONFIG

def _load_thresholds():
    t = CONFIG["thresholds"]
    return {
        container: [
            (lo, hi, status.upper())
            for status, (lo, hi) in t[container].items()
        ]
        for container in ("water", "food")
    }

_THRESHOLDS = _load_thresholds()

def pct_to_status(pct: float, container: str = "water") -> str:
    for lo, hi, status in _THRESHOLDS[container]:
        if lo <= pct < hi:
            return status
    return "CRITICAL"