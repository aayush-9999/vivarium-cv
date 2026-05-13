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
        from core.config import CLASS_TO_LEVEL
        if class_id not in CLASS_TO_LEVEL:
            return cls(pct=0.0, status="CRITICAL")
        _, status, pct = CLASS_TO_LEVEL[class_id]
        return cls(pct=pct, status=status)

    @classmethod
    def unknown(cls) -> "LevelReading":
        return cls(pct=0.0, status="CRITICAL")


class BeddingReading(BaseModel):
    area_pct:  float = Field(..., ge=0.0, le=100.0)
    condition: str   = Field(..., pattern="^(WORST|BAD|OK|PERFECT|NOT_DETECTED)$")  # ← added NOT_DETECTED

    @classmethod
    def from_class_id(cls, class_id: int, area_pct: float = 0.0) -> "BeddingReading":
        mapping = {
            9:  "WORST",
            10: "BAD",
            11: "OK",
            12: "PERFECT",
        }
        condition = mapping.get(class_id, "OK")
        return cls(area_pct=round(area_pct, 2), condition=condition)

    @classmethod
    def not_detected(cls) -> "BeddingReading":          # ← added this
        return cls(area_pct=0.0, condition="NOT_DETECTED")


class DetectionResult(BaseModel):
    cage_id:      str
    timestamp:    datetime
    mouse_count:  int             = Field(..., ge=0)
    water:        LevelReading
    food:         LevelReading
    bedding:      BeddingReading  = Field(
        default_factory=BeddingReading.not_detected,
        description="Bedding cleanliness reading",
    )
    inference_ms: Optional[int]   = None
    image_path:   Optional[str]   = None
    mouse_bboxes: list[BoundingBox] = Field(default_factory=list)
    water_bbox:   Optional[BoundingBox] = None
    food_bbox:    Optional[BoundingBox] = None
    bedding_bbox: Optional[BoundingBox] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}