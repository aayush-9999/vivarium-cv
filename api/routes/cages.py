# api/routes/cages.py
"""
Vivarium cage monitoring endpoints.

These are the only REST endpoints that belong in a production inference
server — querying cage state, history, and alerts.

All ML training / data-pipeline operations have been removed from the API.
Run those via the orchestrator directly or via the CLI scripts in scripts/.

Endpoints
─────────
GET  /cages                         — latest reading for every known cage
GET  /cages/critical                — cages currently in CRITICAL state
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

from db.session import get_db
from db import crud

router = APIRouter(tags=["cages"])


# ─────────────────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────────────────

class CageLatestResponse(BaseModel):
    cage_id:      str
    recorded_at:  datetime
    mouse_count:  Optional[int]
    water_pct:    Optional[float]
    water_status: Optional[str]
    food_pct:     Optional[float]
    food_status:  Optional[str]
    inference_ms: Optional[int]
    image_path:   Optional[str]

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
    """
    Returns the single most-recent reading for every cage_id that has
    ever sent a frame.  Useful for a dashboard overview.
    """
    readings = await crud.get_all_latest_readings(db)
    if not readings:
        return []
    return readings


@router.get(
    "/cages/critical",
    response_model=list[CageLatestResponse],
    summary="Cages currently in CRITICAL state",
)
async def list_critical_cages(db: AsyncSession = Depends(get_db)):
    """
    Returns the latest reading for every cage where water OR food status
    is currently CRITICAL.  Intended as the data source for alert systems —
    poll this endpoint to know which cages need immediate attention.

    Empty list means all cages are OK.
    """
    readings = await crud.get_critical_cages(db)
    return readings


@router.get(
    "/cages/{cage_id}",
    response_model=CageLatestResponse,
    summary="Latest reading for a specific cage",
)
async def get_cage(cage_id: str, db: AsyncSession = Depends(get_db)):
    """
    Returns the most recent DetectionResult stored for this cage_id.
    Raises 404 if the cage has never sent a frame.
    """
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
    limit:   int = Query(default=50, ge=1, le=500, description="Number of readings to return"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the last N readings for a cage, newest first.
    Use this to plot water/food level trends over time.

    - Default: 50 readings
    - Max: 500 readings
    """
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
    Returns every unresolved alert across all cages, newest first.

    Alert types:
    - water_critical  — water level dropped to CRITICAL (0–15%)
    - food_critical   — food level dropped to CRITICAL (0–15%)

    Alerts are auto-created when a detection result comes in with a
    CRITICAL status.  They remain open until resolved via POST /alerts/{cage_id}/resolve.
    """
    alerts = await crud.get_active_alerts(db)
    return alerts


@router.get(
    "/alerts/{cage_id}",
    response_model=list[AlertResponse],
    summary="Alerts for a specific cage",
)
async def get_cage_alerts(
    cage_id:          str,
    include_resolved: bool = Query(default=False, description="Include resolved alerts"),
    limit:            int  = Query(default=20, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns alerts for a specific cage.

    By default returns only active (unresolved) alerts.
    Pass `include_resolved=true` to also see historical resolved alerts.
    """
    alerts = await crud.get_alerts_for_cage(
        db,
        cage_id,
        include_resolved=include_resolved,
        limit=limit,
    )
    return alerts


@router.post(
    "/alerts/{cage_id}/resolve",
    response_model=ResolveResponse,
    summary="Mark alerts as resolved for a cage",
)
async def resolve_cage_alert(
    cage_id:    str,
    alert_type: str = Query(
        ...,
        description="Alert type to resolve: 'water_critical' or 'food_critical'",
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Marks all open alerts of the given type for this cage as resolved.

    Call this after a cage has been refilled / restocked so the alert
    stops appearing in GET /alerts and GET /cages/critical.

    Example:
        POST /alerts/cage_01/resolve?alert_type=water_critical
    """
    valid_types = {"water_critical", "food_critical"}
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