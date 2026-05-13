# pipeline/detectors/yolo/yolo_detector.py
from __future__ import annotations

import time

import cv2
import numpy as np
import torch

from core.config_loader import (
    CONFIG,
    YOLOX_EXP_FILE,
    YOLOX_INPUT_SIZE,
    YOLO_CONF_THRESHOLD,
    YOLO_IOU_THRESHOLD,
)
from core.exceptions import DetectorInitError, InferenceError
from core.schemas import DetectionResult
from pipeline.detectors.base import BaseDetector
from pipeline.detectors.yolo.postprocessor import parse_yolox_results


class YOLODetector(BaseDetector):

    def __init__(self, weights_path: str, device: str = "cpu") -> None:
        super().__init__(weights_path=weights_path, device=device)

    def _load_model(self) -> None:
        try:
            from yolox.exp import get_exp
            from yolox.data.data_augment import ValTransform

            assert CONFIG["yolox"]["num_classes"] == 13, (
                f"Config has num_classes={CONFIG['yolox']['num_classes']}, "
                "expected 13. Check YOLOX_NUM_CLASSES in your .env"
            )

            exp             = get_exp(str(YOLOX_EXP_FILE), exp_name=None)
            exp.num_classes = CONFIG["yolox"]["num_classes"]

            self.model = exp.get_model()
            self.model.eval()

            ckpt = torch.load(
                self.weights_path, map_location="cpu", weights_only=False
            )
            self.model.load_state_dict(ckpt.get("model", ckpt))
            self.model.to(self.device)

            self._preproc    = ValTransform(legacy=False)
            self._input_size = YOLOX_INPUT_SIZE

        except AssertionError:
            raise
        except FileNotFoundError as exc:
            raise DetectorInitError(
                f"YOLOX weights not found: {self.weights_path}"
            ) from exc
        except Exception as exc:
            raise DetectorInitError(
                f"Failed to load YOLOX model: {exc}"
            ) from exc

    def detect(self, frame: np.ndarray, cage_id: str) -> DetectionResult:
        if not self.is_ready():
            raise InferenceError("Model not loaded — call _load_model() first.")

        t_start   = time.perf_counter_ns()
        orig_size = frame.shape[:2]          # ← ADD: capture (h, w) before preprocessing

        try:
            img, ratio = self._preproc(frame, None, self._input_size)
            img_tensor = torch.from_numpy(img).unsqueeze(0).float()
            if self.device != "cpu":
                img_tensor = img_tensor.cuda()

            with torch.no_grad():
                raw_output = self.model(img_tensor)

            from yolox.utils import postprocess
            outputs = postprocess(
                raw_output,
                num_classes = CONFIG["yolox"]["num_classes"],
                conf_thre   = YOLO_CONF_THRESHOLD,
                nms_thre    = YOLO_IOU_THRESHOLD,
            )

        except Exception as exc:
            raise InferenceError(
                f"YOLOX inference failed for cage '{cage_id}': {exc}"
            ) from exc

        return self._postprocess((outputs, ratio), cage_id, t_start, orig_size)  # ← pass orig_size


    def _postprocess(
        self,
        raw_output,
        cage_id: str,
        inference_start_ns: int = 0,
        orig_size: tuple[int, int] = None,   # ← ADD parameter
    ) -> DetectionResult:
        outputs, ratio = raw_output
        return parse_yolox_results(
            outputs            = outputs,
            ratio              = ratio,
            cage_id            = cage_id,
            inference_start_ns = inference_start_ns,
            input_size         = self._input_size,   # ← ADD
            orig_size          = orig_size,           # ← ADD
        )

    