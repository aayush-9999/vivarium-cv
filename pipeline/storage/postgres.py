# pipeline/storage/postgres.py  — BEDDING PATCH
"""
CRUD helpers.

BEDDING PATCH
─────────────
save_detection() now:
  - writes bedding_area_pct / bedding_condition columns
  - auto-creates a "bedding_bad" Alert when condition == "BAD"

resolve_alert() is unchanged — callers pass alert_type="bedding_bad".
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, desc, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.storage.models import CageReading, Alert
from core.schemas import DetectionResult


# ─────────────────────────────────────────────────────────────────────────────
# Writes
# ─────────────────────────────────────────────────────────────────────────────

async def save_detection(db: AsyncSession, result: DetectionResult) -> None:
    """Persist a DetectionResult to the cage_readings table."""
    row = CageReading(
        cage_id           = result.cage_id,
        recorded_at       = result.timestamp,
        mouse_count       = result.mouse_count,
        water_pct         = result.water.pct,
        water_status      = result.water.status,
        food_pct          = result.food.pct,
        food_status       = result.food.status,
        # ── Bedding ───────────────────────────────────────────────────────
        bedding_area_pct  = result.bedding.area_pct,
        bedding_condition = result.bedding.condition,
        # ─────────────────────────────────────────────────────────────────
        inference_ms      = result.inference_ms,
        image_path        = result.image_path,
    )
    db.add(row)

    # Auto-create alerts for CRITICAL water/food AND BAD bedding
    alert_checks = [
        ("water_critical", result.water.status == "CRITICAL"),
        ("food_critical",  result.food.status  == "CRITICAL"),
        ("bedding_bad",    result.bedding.condition == "BAD"),    # ← new
    ]

    for alert_type, triggered in alert_checks:
        if not triggered:
            continue
        existing = await db.scalar(
            select(Alert).where(
                and_(
                    Alert.cage_id    == result.cage_id,
                    Alert.alert_type == alert_type,
                    Alert.resolved_at.is_(None),
                )
            )
        )
        if not existing:
            db.add(Alert(
                cage_id      = result.cage_id,
                alert_type   = alert_type,
                triggered_at = result.timestamp,
                resolved_at  = None,
                notified     = False,
            ))

    await db.commit()


async def resolve_alert(db: AsyncSession, cage_id: str, alert_type: str) -> bool:
    """
    Mark all open alerts of a given type for a cage as resolved.
    Returns True if at least one alert was resolved.

    Valid alert_type values: water_critical | food_critical | bedding_bad
    """
    result = await db.execute(
        select(Alert).where(
            and_(
                Alert.cage_id    == cage_id,
                Alert.alert_type == alert_type,
                Alert.resolved_at.is_(None),
            )
        )
    )
    alerts = result.scalars().all()
    if not alerts:
        return False
    now = datetime.now(tz=timezone.utc)
    for alert in alerts:
        alert.resolved_at = now
    await db.commit()
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Cage reads
# ─────────────────────────────────────────────────────────────────────────────

async def get_latest_reading(
    db: AsyncSession,
    cage_id: str,
) -> Optional[CageReading]:
    return await db.scalar(
        select(CageReading)
        .where(CageReading.cage_id == cage_id)
        .order_by(desc(CageReading.recorded_at))
        .limit(1)
    )


async def get_cage_history(
    db: AsyncSession,
    cage_id: str,
    limit: int = 50,
) -> list[CageReading]:
    result = await db.execute(
        select(CageReading)
        .where(CageReading.cage_id == cage_id)
        .order_by(desc(CageReading.recorded_at))
        .limit(limit)
    )
    return result.scalars().all()


async def get_all_latest_readings(db: AsyncSession) -> list[CageReading]:
    subq = (
        select(
            CageReading.cage_id,
            func.max(CageReading.recorded_at).label("max_ts"),
        )
        .group_by(CageReading.cage_id)
        .subquery()
    )
    result = await db.execute(
        select(CageReading).join(
            subq,
            and_(
                CageReading.cage_id     == subq.c.cage_id,
                CageReading.recorded_at == subq.c.max_ts,
            ),
        )
    )
    return result.scalars().all()


async def get_critical_cages(db: AsyncSession) -> list[CageReading]:
    """
    Latest reading for every cage where water/food is CRITICAL **or**
    bedding condition is BAD.
    """
    all_latest = await get_all_latest_readings(db)
    return [
        r for r in all_latest
        if r.water_status      == "CRITICAL"
        or r.food_status       == "CRITICAL"
        or r.bedding_condition == "BAD"        # ← new
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Alert reads (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

async def get_active_alerts(db: AsyncSession) -> list[Alert]:
    result = await db.execute(
        select(Alert)
        .where(Alert.resolved_at.is_(None))
        .order_by(desc(Alert.triggered_at))
    )
    return result.scalars().all()


async def get_alerts_for_cage(
    db: AsyncSession,
    cage_id: str,
    include_resolved: bool = False,
    limit: int = 20,
) -> list[Alert]:
    filters = [Alert.cage_id == cage_id]
    if not include_resolved:
        filters.append(Alert.resolved_at.is_(None))
    result = await db.execute(
        select(Alert)
        .where(and_(*filters))
        .order_by(desc(Alert.triggered_at))
        .limit(limit)
    )
    return result.scalars().all()