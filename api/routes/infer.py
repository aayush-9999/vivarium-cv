# api/routes/infer.py
"""
POST /infer — accepts a JPEG/PNG frame upload and cage_id,
runs the full YOLO pipeline, persists to DB, returns DetectionResult.
"""
from __future__ import annotations
import numpy as np  
import cv2

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.schemas import DetectionResult
from core.exceptions import VivariumCVError
from pipeline.pipeline_factory import get_pipeline
from db.session import get_db         
from db.crud import save_detection     # thin write helper
import logging
logger = logging.getLogger("vivarium.api")

router = APIRouter(prefix="/infer", tags=["inference"])

# Module-level pipeline singleton — loaded once at startup
# (FastAPI's lifespan event is the cleaner approach, but this works for now)
_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = get_pipeline()
    return _pipeline


@router.post("/", response_model=DetectionResult)
async def infer(
    cage_id: str        = Form(..., description="Cage identifier, e.g. 'cage_01'"),
    frame:   UploadFile = File(..., description="JPEG or PNG frame from camera"),
    save_flagged: bool  = Form(False, description="Save frame to disk if CRITICAL"),
    db: AsyncSession    = Depends(get_db),
):
    """
    Run mouse detection + water/food level estimation on an uploaded frame.

    - Accepts multipart/form-data with `cage_id` + `frame` file.
    - Returns a `DetectionResult` JSON object.
    - Persists reading to `cage_readings` table.
    """
    # ── Decode uploaded image ──────────────────────────────────────
    raw_bytes = await frame.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded frame is empty.")

    np_arr    = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if bgr_frame is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode image. Ensure the file is a valid JPEG or PNG.",
        )

    # ── Run pipeline ───────────────────────────────────────────────
    try:
        pipeline = _get_pipeline()
        result = pipeline.run(
            frame=bgr_frame,
            cage_id=cage_id,
            save_flagged=True,
        )
    except VivariumCVError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ── Persist to DB ──────────────────────────────────────────────
    try:
        await save_detection(db, result)
    except Exception as e:
        logger.error("DB write failed for cage '%s': %s", cage_id, e)
        # inference succeeded — return result even if DB write failed

    return result