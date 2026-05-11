# pipeline/notifiers/webhook.py
import logging
import requests
from pipeline.notifiers.base import BaseNotifier

logger = logging.getLogger("vivarium.notifiers.webhook")

class WebhookNotifier(BaseNotifier):
    def __init__(self, url: str):
        self.url = url

    def notify(self, cage_id: str, alert_type: str, pct: float) -> None:
        payload = {"cage_id": cage_id, "alert_type": alert_type, "pct": pct}
        try:
            resp = requests.post(self.url, json=payload, timeout=5)
            resp.raise_for_status()
            logger.info("Webhook sent for %s/%s", cage_id, alert_type)
        except Exception as e:
            logger.error("Webhook failed: %s", e)