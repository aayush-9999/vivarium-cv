# pipeline/measurers/factory.py
import os
from core.config_loader import CONFIG

def get_measurer():
    from pipeline.measurers.pspnet_measurer import LevelEstimator
    return LevelEstimator(
        water_weights=CONFIG["pspnet"]["water_weights"],
        food_weights=CONFIG["pspnet"]["food_weights"],
        backbone=CONFIG["pspnet"]["backbone"],
        device=CONFIG["device"],
    )