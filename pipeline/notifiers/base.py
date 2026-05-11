# pipeline/notifiers/base.py
from abc import ABC, abstractmethod

class BaseNotifier(ABC):
    @abstractmethod
    def notify(self, cage_id: str, alert_type: str, pct: float) -> None:
        pass