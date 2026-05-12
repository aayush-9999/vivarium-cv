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


class BeddingReading(BaseModel):
    """
    Bedding cleanliness result derived from the bedding YOLOX detection.

    area_pct : estimated fraction of the cage floor covered by bedding (0–100).
               Computed as (bbox_area / frame_area) * 100 from the YOLOX bbox.
               When multiple bedding boxes are found the sum of their areas is used
               (capped at 100).
    condition : "GOOD"  — area_pct < 50  → bedding occupies less than half the floor,
                           still clean / not soiled.
               "BAD"   — area_pct >= 50 → bedding covers >= 50 % of the floor,
                           likely soiled / needs changing.
    """
    area_pct:  float = Field(..., ge=0.0, le=100.0, description="Bedding bbox area % of frame")
    condition: str   = Field(..., pattern="^(GOOD|BAD)$")

    @classmethod
    def from_area_pct(cls, area_pct: float) -> "BeddingReading":
        condition = "BAD" if area_pct >= 50.0 else "GOOD"
        return cls(area_pct=round(area_pct, 2), condition=condition)

    @classmethod
    def not_detected(cls) -> "BeddingReading":
        """No bedding box found — treat as GOOD (assume clean / empty frame)."""
        return cls(area_pct=0.0, condition="GOOD")


class DetectionResult(BaseModel):
    cage_id:      str
    timestamp:    datetime
    mouse_count:  int             = Field(..., ge=0)
    water:        LevelReading
    food:         LevelReading
    bedding:      BeddingReading  = Field(
        default_factory=BeddingReading.not_detected,
        description="Bedding cleanliness reading (GOOD / BAD)",
    )
    inference_ms: Optional[int]   = None
    image_path:   Optional[str]   = None
    water_bbox:   Optional[BoundingBox] = None
    food_bbox:    Optional[BoundingBox] = None
    bedding_bbox: Optional[BoundingBox] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
