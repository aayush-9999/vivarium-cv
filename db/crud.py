# db/crud.py
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import CageReading
from core.schemas import DetectionResult


async def save_detection(db: AsyncSession, result: DetectionResult) -> None:
    """Persist a DetectionResult to the cage_readings table."""
    row = CageReading(
        cage_id      = result.cage_id,
        recorded_at  = result.timestamp,
        mouse_count  = result.mouse_count,
        water_pct    = result.water.pct,
        water_status = result.water.status,
        food_pct     = result.food.pct,
        food_status  = result.food.status,
        inference_ms = result.inference_ms,
        image_path   = result.image_path,
    )
    db.add(row)
    await db.commit()