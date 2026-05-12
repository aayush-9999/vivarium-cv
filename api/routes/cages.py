# api/routes/cages.py  — BEDDING PATCH
"""
Vivarium cage monitoring endpoints.

BEDDING PATCH
─────────────
• CageLatestResponse now includes bedding_area_pct / bedding_condition.
• GET /cages/critical now returns cages with bedding_condition="BAD" too.
• POST /alerts/{cage_id}/resolve accepts alert_type="bedding_bad".

Endpoints
─────────
GET  /cages                         — latest reading for every known cage
GET  /cages/critical                — cages with CRITICAL water/food **or** BAD bedding
GET  /cages/{cage_id}               — latest reading for one cage
GET  /cages/{cage_id}/history       — recent reading history for one cage

GET  /alerts                        — all active (unresolved) alerts
GET  /alerts/{cage_id}              — active alerts for one cage
POST /alerts/{cage_id}/resolve      — mark a cage's alerts as resolved
"""

from __future__ import annotations

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from pipeline.storage import postgres as crud
from pipeline.storage.session import get_db

router = APIRouter(tags=["cages"])


# ─────────────────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────────────────

class CageLatestResponse(BaseModel):
    cage_id:           str
    recorded_at:       datetime
    mouse_count:       Optional[int]
    water_pct:         Optional[float]
    water_status:      Optional[str]
    food_pct:          Optional[float]
    food_status:       Optional[str]
    # ── Bedding ───────────────────────────────────────────────────────────
    bedding_area_pct:  Optional[float]   # 0–100 (% of frame)
    bedding_condition: Optional[str]     # GOOD | BAD
    # ─────────────────────────────────────────────────────────────────────
    inference_ms:      Optional[int]
    image_path:        Optional[str]

    class Config:
        from_attributes = True


class AlertResponse(BaseModel):
    id:           int
    cage_id:      str
    alert_type:   str
    triggered_at: datetime
    resolved_at:  Optional[datetime]
    notified:     bool

    class Config:
        from_attributes = True


class ResolveResponse(BaseModel):
    cage_id:    str
    alert_type: str
    resolved:   bool
    message:    str


# ─────────────────────────────────────────────────────────────────────────────
# Cage endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/cages",
    response_model=list[CageLatestResponse],
    summary="Latest reading for every known cage",
)
async def list_cages(db: AsyncSession = Depends(get_db)):
    readings = await crud.get_all_latest_readings(db)
    return readings or []


@router.get(
    "/cages/critical",
    response_model=list[CageLatestResponse],
    summary="Cages currently in CRITICAL state or with BAD bedding",
)
async def list_critical_cages(db: AsyncSession = Depends(get_db)):
    """
    Returns the latest reading for every cage where:
      - water OR food status is CRITICAL, **or**
      - bedding condition is BAD (area_pct >= 50 %).

    An empty list means all cages are OK.
    """
    return await crud.get_critical_cages(db)


@router.get(
    "/cages/{cage_id}",
    response_model=CageLatestResponse,
    summary="Latest reading for a specific cage",
)
async def get_cage(cage_id: str, db: AsyncSession = Depends(get_db)):
    reading = await crud.get_latest_reading(db, cage_id)
    if reading is None:
        raise HTTPException(
            status_code=404,
            detail=f"No readings found for cage '{cage_id}'.",
        )
    return reading


@router.get(
    "/cages/{cage_id}/history",
    response_model=list[CageLatestResponse],
    summary="Recent reading history for a specific cage",
)
async def get_cage_history(
    cage_id: str,
    limit:   int = Query(default=50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    readings = await crud.get_cage_history(db, cage_id, limit=limit)
    if not readings:
        raise HTTPException(
            status_code=404,
            detail=f"No readings found for cage '{cage_id}'.",
        )
    return readings


# ─────────────────────────────────────────────────────────────────────────────
# Alert endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/alerts",
    response_model=list[AlertResponse],
    summary="All active unresolved alerts across all cages",
)
async def list_active_alerts(db: AsyncSession = Depends(get_db)):
    """
    Alert types:
    - water_critical — water level dropped to CRITICAL (0–15 %)
    - food_critical  — food level dropped to CRITICAL (0–15 %)
    - bedding_bad    — bedding area >= 50 % of frame (needs changing)
    """
    return await crud.get_active_alerts(db)


@router.get(
    "/alerts/{cage_id}",
    response_model=list[AlertResponse],
    summary="Alerts for a specific cage",
)
async def get_cage_alerts(
    cage_id:          str,
    include_resolved: bool = Query(default=False),
    limit:            int  = Query(default=20, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    return await crud.get_alerts_for_cage(
        db, cage_id, include_resolved=include_resolved, limit=limit,
    )


@router.post(
    "/alerts/{cage_id}/resolve",
    response_model=ResolveResponse,
    summary="Mark alerts as resolved for a cage",
)
async def resolve_cage_alert(
    cage_id:    str,
    alert_type: str = Query(
        ...,
        description=(
            "Alert type to resolve: "
            "'water_critical' | 'food_critical' | 'bedding_bad'"
        ),
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Marks all open alerts of the given type for this cage as resolved.

    Examples:
        POST /alerts/cage_01/resolve?alert_type=water_critical
        POST /alerts/cage_01/resolve?alert_type=bedding_bad
    """
    valid_types = {"water_critical", "food_critical", "bedding_bad"}
    if alert_type not in valid_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid alert_type '{alert_type}'. Must be one of: {sorted(valid_types)}",
        )

    resolved = await crud.resolve_alert(db, cage_id, alert_type)

    if not resolved:
        return ResolveResponse(
            cage_id=cage_id,
            alert_type=alert_type,
            resolved=False,
            message=f"No open '{alert_type}' alert found for cage '{cage_id}'.",
        )

    return ResolveResponse(
        cage_id=cage_id,
        alert_type=alert_type,
        resolved=True,
        message=f"All open '{alert_type}' alerts for cage '{cage_id}' have been resolved.",
    )