# pipeline/threshold/cooldown.py
"""
Prevents alert spam by enforcing a minimum interval between
alerts of the same type for the same cage.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Dict, Tuple

class CooldownManager:
    def __init__(self, cooldown_seconds: int = 300):
        self.cooldown_seconds = cooldown_seconds
        self._last_alert: Dict[Tuple[str, str], datetime] = {}

    def should_alert(self, cage_id: str, alert_type: str) -> bool:
        key = (cage_id, alert_type)
        now = datetime.now(tz=timezone.utc)
        last = self._last_alert.get(key)
        if last is None:
            self._last_alert[key] = now
            return True
        elapsed = (now - last).total_seconds()
        if elapsed >= self.cooldown_seconds:
            self._last_alert[key] = now
            return True
        return False

    def reset(self, cage_id: str, alert_type: str) -> None:
        self._last_alert.pop((cage_id, alert_type), None)   