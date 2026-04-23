# core/schemas.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class BoundingBox(BaseModel):
    x1:   float
    y1:   float
    x2:   float
    y2:   float
    conf: float = Field(..., ge=0.0, le=1.0)


class LevelReading(BaseModel):
    pct:    float = Field(..., ge=0.0, le=100.0)
    status: str   = Field(..., pattern="^(OK|LOW|CRITICAL)$")

    @classmethod
    def from_class_id(cls, class_id: int) -> "LevelReading":
        """
        Build a LevelReading directly from a YOLO class ID (1-8).
        No HSV, no colour math — the class IS the level.

        class_id  type    status     pct
        ────────  ──────  ─────────  ────
        1         water   CRITICAL    7.5
        2         water   LOW        25.0
        3         water   OK         57.5
        4         water   OK         90.0
        5         food    CRITICAL    7.5
        6         food    LOW        25.0
        7         food    OK         57.5
        8         food    OK         90.0
        """
        from core.config import CLASS_TO_LEVEL
        if class_id not in CLASS_TO_LEVEL:
            return cls(pct=0.0, status="CRITICAL")
        _, status, pct = CLASS_TO_LEVEL[class_id]
        return cls(pct=pct, status=status)

    @classmethod
    def unknown(cls) -> "LevelReading":
        """Used when YOLO didn't detect any container at all."""
        return cls(pct=0.0, status="CRITICAL")


class DetectionResult(BaseModel):
    cage_id:      str
    timestamp:    datetime
    mouse_count:  int             = Field(..., ge=0)
    water:        LevelReading
    food:         LevelReading
    inference_ms: Optional[int]   = None
    image_path:   Optional[str]   = None
    water_bbox:   Optional[BoundingBox] = None
    food_bbox:    Optional[BoundingBox] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

