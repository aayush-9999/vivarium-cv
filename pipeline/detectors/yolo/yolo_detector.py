# detectors/yolo/yolo_detector.py
from __future__ import annotations
import time
from unittest import result
import numpy as np
import torch
import cv2

from core.base_detector import BaseDetector
from core.schemas import DetectionResult
from core.config import YOLOX_WEIGHTS, YOLOX_EXP_FILE, YOLOX_INPUT_SIZE, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD
from core.exceptions import DetectorInitError, InferenceError
from pipeline.detectors.yolo.postprocessor import parse_yolox_results


class YOLODetector(BaseDetector):

    def __init__(self, weights_path: str, device: str = "cpu"):
        super().__init__(weights_path=weights_path, device=device)

    def _load_model(self) -> None:
        try:
            from yolox.exp import get_exp
            from yolox.utils import get_model_info, postprocess
            from yolox.data.data_augment import ValTransform

            exp = get_exp(str(YOLOX_EXP_FILE), exp_name=None)
            exp.num_classes = 9

            self.model = exp.get_model()
            self.model.eval()

            # Load checkpoint — YOLOx uses {"model": state_dict} format
            ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            print("🔥 ACTUAL WEIGHTS LOADED:", self.weights_path)
            self.model.load_state_dict(ckpt.get("model", ckpt))
            self.model.to(self.device)

            # YOLOx uses its own preprocessing transform
            self._preproc = ValTransform(legacy=False)
            self._input_size = YOLOX_INPUT_SIZE  # (h, w)

        except FileNotFoundError as e:
            raise DetectorInitError(f"YOLOx weights not found: {self.weights_path}") from e
        except Exception as e:
            raise DetectorInitError(f"Failed to load YOLOx model: {e}") from e

    def detect(self, frame: np.ndarray, cage_id: str) -> DetectionResult:
        if not self.is_ready():
            raise InferenceError("Model not loaded.")

        t_start = time.perf_counter_ns()

        try:
            # YOLOx preprocessing: outputs (img, ratio) not just img
            img, ratio = self._preproc(frame, None, self._input_size)
            img_tensor = torch.from_numpy(img).unsqueeze(0).float()
            if self.device != "cpu":
                img_tensor = img_tensor.cuda()

            with torch.no_grad():
                raw_output = self.model(img_tensor)

            # YOLOx postprocess applies NMS internally
            from yolox.utils import postprocess
            outputs = postprocess(
                raw_output,
                num_classes=9,
                conf_thre=YOLO_CONF_THRESHOLD,
                nms_thre=YOLO_IOU_THRESHOLD,
            )

        except Exception as e:
            raise InferenceError(f"YOLOx inference failed for cage '{cage_id}': {e}") from e

        result = self._postprocess((outputs, ratio), cage_id, t_start)
        return result


    def _postprocess(self, raw_output, cage_id: str, inference_start_ns: int = 0) -> DetectionResult:
        outputs, ratio = raw_output
        return parse_yolox_results(
            outputs=outputs,
            ratio=ratio,
            cage_id=cage_id,
            inference_start_ns=inference_start_ns,
        )

    def warmup(self) -> None:
        dummy = np.zeros((*YOLOX_INPUT_SIZE, 3), dtype=np.uint8)
        self.detect(dummy, cage_id="__warmup__")