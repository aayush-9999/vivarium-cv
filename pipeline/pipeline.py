# pipeline/pipeline.py

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from core.config_loader import CONFIG
from core.exceptions import VivariumCVError
from core.schemas import DetectionResult
from pipeline.annotator.factory import get_annotator
from pipeline.detectors.factory import get_detector
from pipeline.preprocessors.background_subtractor import BackgroundSubtractor
from pipeline.preprocessors.frame_preprocessor import FramePreprocessor

logger = logging.getLogger("vivarium.pipeline")


class InferencePipeline:

    def __init__(self, cage_type: str = "default", backend: Optional[str] = None) -> None:
        self._backend   = (backend or CONFIG["backend"]).lower()
        self._cage_type = cage_type
        self._annotator = get_annotator()

        self._preprocessor  = FramePreprocessor(cage_type=cage_type)
        self._bg_subtractor = BackgroundSubtractor()
        self._detector      = get_detector(cage_type=cage_type, backend=self._backend)

        self._detector.warmup()
        logger.info("InferencePipeline ready — backend=%s  cage=%s", self._backend, cage_type)

    def run(
        self,
        frame: np.ndarray,
        cage_id: str,
        save_flagged: bool = False,
        output_dir: str = "flagged_frames",
    ) -> DetectionResult:
        if frame is None or frame.size == 0:
            raise VivariumCVError(f"Empty frame for cage '{cage_id}'.")

        result = self._detector.detect(frame=frame, cage_id=cage_id)

        if save_flagged and _is_critical(result):
            annotated  = self._annotator.draw(frame, result)
            image_path = _save_frame(annotated, cage_id, output_dir)
            result     = result.model_copy(update={"image_path": image_path})

        return result

    def set_reference_frame(self, frame: np.ndarray) -> None:
        self._bg_subtractor.set_reference(self._preprocessor.resize(frame))

    def has_motion(self, frame: np.ndarray) -> bool:
        if not self._bg_subtractor.has_reference():
            return True
        return self._bg_subtractor.has_motion(self._preprocessor.resize(frame))

    def debug_frame(self, frame: np.ndarray) -> np.ndarray:
        result = self._detector.detect(frame=frame, cage_id="__debug__")
        return self._annotator.draw(frame, result)

    @property
    def backend(self) -> str:
        return self._backend


def _is_critical(result: DetectionResult) -> bool:
    return result.water.status == "CRITICAL" or result.food.status == "CRITICAL" or result.bedding.condition in ("BAD", "WORST")


def _save_frame(frame: np.ndarray, cage_id: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(output_dir, f"{cage_id}_{ts}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path