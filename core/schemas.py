# core/schemas.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Tuple

class BoundingBox(BaseModel):
    """Normalised or pixel-space bounding box from YOLO."""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = Field(..., ge=0.0, le=1.0)

class LevelReading(BaseModel):
    pct:    float  = Field(..., ge=0.0, le=100.0)
    status: str    = Field(..., pattern="^(OK|LOW|CRITICAL)$")

class DetectionResult(BaseModel):
    cage_id:        str
    timestamp:      datetime
    mouse_count:    int              = Field(..., ge=0)
    water:          LevelReading
    food:           LevelReading
    inference_ms:   Optional[int]    = None
    image_path:     Optional[str]    = None

    # YOLO-detected container bounding boxes (pixel coords in 640x640 space)
    # None means YOLO didn't detect that container — pipeline falls back to hardcoded ROI
    water_bbox:     Optional[BoundingBox] = None
    food_bbox:      Optional[BoundingBox] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class CageStatus(BaseModel):
    cage_id:      str
    last_updated: datetime
    water:        LevelReading
    food:         LevelReading
    mouse_count:  int
    is_stale:     bool = False  # True if last reading > 15 min old

class AlertEvent(BaseModel):
    cage_id:      str
    alert_type:   str       # WATER_LOW | WATER_CRITICAL | FOOD_LOW | FOOD_CRITICAL
    triggered_at: datetime
    resolved_at:  Optional[datetime] = None
    notified:     bool = False
