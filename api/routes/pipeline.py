# api/routes/pipeline.py
"""
POST /pipeline/* — HTTP endpoints for all pipeline operations.

Every workflow previously run as a script is now accessible
via REST API, in addition to the Python orchestrator.

Endpoints:
    POST /pipeline/infer          — inference on uploaded frame
    POST /pipeline/label          — run GDINO labelling
    POST /pipeline/augment        — augment dataset
    POST /pipeline/clean          — clean food labels
    POST /pipeline/dedup          — dedup labels
    POST /pipeline/split          — split dataset
    POST /pipeline/train          — start training run
    POST /pipeline/validate       — validate on val split
    POST /pipeline/run-data       — full data prep pipeline
    POST /pipeline/run-full       — end-to-end data + train + val
    GET  /pipeline/verify-labels  — verify label integrity
"""

from __future__ import annotations

import io
import numpy as np
import cv2

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

from core.schemas import DetectionResult
from core.exceptions import VivariumCVError
from pipeline.pipeline_factory import get_orchestrator
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from db.session import get_db
from db.crud import save_detection

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# Singleton orchestrator
_orch = None

def _get_orch():
    global _orch
    if _orch is None:
        _orch = get_orchestrator()
    return _orch


# ── Request / Response models ──────────────────────────────────────────────────

class LabelRequest(BaseModel):
    propagate:        bool  = False
    mouse_thresh:     float = 0.22
    container_thresh: float = 0.30
    food_thresh:      float = 0.28

class AugmentRequest(BaseModel):
    n:            int  = 50
    seed:         Optional[int] = None
    jpeg_quality: int  = 90

class TrainRequest(BaseModel):
    epochs:   int  = 100
    batch:    int  = 16
    device:   str  = "cpu"
    run_name: str  = "vivarium_v1"
    resume:   bool = False

class ValidateRequest(BaseModel):
    conf: float = 0.45
    iou:  float = 0.30

class DataPipelineRequest(BaseModel):
    n_augments: int  = 50
    propagate:  bool = True

class CleanRequest(BaseModel):
    max_area: float = 0.12
    max_w:    float = 0.40
    max_h:    float = 0.50
    dry_run:  bool  = False

class DedupRequest(BaseModel):
    mouse_iou: float = 0.45
    food_iou:  float = 0.50
    dry_run:   bool  = False

class SplitRequest(BaseModel):
    train_ratio: float = 0.85
    seed:        int   = 42

class StatusResponse(BaseModel):
    status:  str
    message: str
    data:    Optional[dict] = None


# ── Inference ─────────────────────────────────────────────────────────────────

@router.post("/infer", response_model=DetectionResult)
async def pipeline_infer(
    cage_id:      str        = Form(...),
    frame:        UploadFile = File(...),
    save_flagged: bool       = Form(False),
    db: AsyncSession         = Depends(get_db),
):
    """
    Run the full inference pipeline on an uploaded frame.
    Identical to POST /infer but routed through the orchestrator.
    """
    raw = await frame.read()
    if not raw:
        raise HTTPException(400, "Uploaded frame is empty.")

    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(422, "Cannot decode image — must be JPEG or PNG.")

    try:
        result = _get_orch().infer(bgr, cage_id=cage_id, save_flagged=save_flagged)
        await save_detection(db, result)
        return result
    except VivariumCVError as e:
        raise HTTPException(500, str(e))


# ── Data labelling ────────────────────────────────────────────────────────────

@router.post("/label", response_model=StatusResponse)
async def pipeline_label(req: LabelRequest, background_tasks: BackgroundTasks):
    """
    Run GDINO labelling on original images (runs in background).
    Check server logs for per-image progress.
    """
    def _run():
        _get_orch().label_originals(
            propagate=req.propagate,
            mouse_thresh=req.mouse_thresh,
            container_thresh=req.container_thresh,
            food_thresh=req.food_thresh,
        )

    background_tasks.add_task(_run)
    return StatusResponse(
        status="started",
        message="GDINO labelling started in background. Check server logs for progress.",
    )


# ── Augmentation ──────────────────────────────────────────────────────────────

@router.post("/augment", response_model=StatusResponse)
async def pipeline_augment(req: AugmentRequest, background_tasks: BackgroundTasks):
    """Augment dataset (runs in background)."""
    def _run():
        _get_orch().augment(n=req.n, seed=req.seed, jpeg_quality=req.jpeg_quality)

    background_tasks.add_task(_run)
    return StatusResponse(
        status="started",
        message=f"Augmentation started ({req.n} variants per source image).",
    )


# ── Label hygiene ─────────────────────────────────────────────────────────────

@router.post("/clean", response_model=StatusResponse)
async def pipeline_clean(req: CleanRequest):
    """Remove oversized food_area label boxes."""
    _get_orch().clean_food_labels(
        max_area=req.max_area,
        max_w=req.max_w,
        max_h=req.max_h,
        dry_run=req.dry_run,
    )
    return StatusResponse(
        status="ok",
        message=f"Food label cleaning {'(dry-run)' if req.dry_run else 'complete'}.",
    )


@router.post("/dedup", response_model=StatusResponse)
async def pipeline_dedup(req: DedupRequest):
    """Deduplicate labels per class rules."""
    _get_orch().dedup_labels(
        mouse_iou=req.mouse_iou,
        food_iou=req.food_iou,
        dry_run=req.dry_run,
    )
    return StatusResponse(
        status="ok",
        message=f"Label dedup {'(dry-run)' if req.dry_run else 'complete'}.",
    )


# ── Dataset split ─────────────────────────────────────────────────────────────

@router.post("/split", response_model=StatusResponse)
async def pipeline_split(req: SplitRequest):
    """Split augmented dataset into train/val."""
    counts = _get_orch().split_dataset(
        train_ratio=req.train_ratio,
        seed=req.seed,
    )
    return StatusResponse(
        status="ok",
        message="Dataset split complete.",
        data=counts,
    )


# ── Training ──────────────────────────────────────────────────────────────────

@router.post("/train", response_model=StatusResponse)
async def pipeline_train(req: TrainRequest, background_tasks: BackgroundTasks):
    """Start a YOLOv8 training run (runs in background)."""
    def _run():
        _get_orch().train(
            epochs=req.epochs,
            batch=req.batch,
            device=req.device,
            run_name=req.run_name,
            resume=req.resume,
        )

    background_tasks.add_task(_run)
    return StatusResponse(
        status="started",
        message=f"Training started: {req.run_name} ({req.epochs} epochs, device={req.device}).",
    )


# ── Validation ────────────────────────────────────────────────────────────────

@router.post("/validate", response_model=StatusResponse)
async def pipeline_validate(req: ValidateRequest, background_tasks: BackgroundTasks):
    """Run inference on val split and save annotated outputs (runs in background)."""
    def _run():
        _get_orch().validate(conf=req.conf, iou=req.iou)

    background_tasks.add_task(_run)
    return StatusResponse(
        status="started",
        message="Validation started. Check runs/val_test/ for outputs.",
    )


# ── Label verification ────────────────────────────────────────────────────────

@router.get("/verify-labels", response_model=StatusResponse)
async def pipeline_verify_labels():
    """Verify label/image pairing and coordinate validity."""
    issues = _get_orch().verify_labels()
    if issues:
        return StatusResponse(
            status="issues_found",
            message=f"{len(issues)} label issues found.",
            data={"issues": issues},
        )
    return StatusResponse(
        status="ok",
        message="All label files are clean.",
    )


# ── Full workflow shortcuts ───────────────────────────────────────────────────

@router.post("/run-data", response_model=StatusResponse)
async def pipeline_run_data(req: DataPipelineRequest, background_tasks: BackgroundTasks):
    """
    Full data preparation pipeline:
    label → augment → clean → dedup → split (runs in background).
    """
    def _run():
        _get_orch().run_data_pipeline(
            n_augments=req.n_augments,
            propagate=req.propagate,
        )

    background_tasks.add_task(_run)
    return StatusResponse(
        status="started",
        message="Full data pipeline started. Check server logs for progress.",
    )


@router.post("/run-full", response_model=StatusResponse)
async def pipeline_run_full(
    n_augments: int = Form(50),
    epochs:     int = Form(100),
    background_tasks: BackgroundTasks = None,
):
    """
    End-to-end pipeline: data prep → train → validate (runs in background).
    This can take hours depending on dataset size and hardware.
    """
    def _run():
        _get_orch().run_full_pipeline(n_augments=n_augments, epochs=epochs)

    background_tasks.add_task(_run)
    return StatusResponse(
        status="started",
        message=f"Full pipeline started ({n_augments} augments, {epochs} epochs). "
                "This will take a while — check server logs.",
    )