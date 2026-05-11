# pipeline/notifiers/factory.py
import os
from typing import List
from pipeline.notifiers.base import BaseNotifier

def get_notifiers() -> List[BaseNotifier]:
    notifiers = []
    webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    if webhook_url:
        from pipeline.notifiers.webhook import WebhookNotifier
        notifiers.append(WebhookNotifier(url=webhook_url))
    return notifiers