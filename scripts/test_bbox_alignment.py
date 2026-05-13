import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ← add this line

import cv2
import numpy as np
from pipeline.pipeline_factory import get_pipeline

img_path = r"C:\Users\aayus\Downloads\IMG_01.png"
img      = cv2.imread(img_path)
print(f"Original frame size: {img.shape}")  # (h, w, 3)

pipeline = get_pipeline()
result   = pipeline._detector.detect(img, cage_id="test")

# Draw all bboxes on original frame
viz = img.copy()
for bbox in result.mouse_bboxes:
    cv2.rectangle(viz,
        (int(bbox.x1), int(bbox.y1)),
        (int(bbox.x2), int(bbox.y2)),
        (255, 220, 0), 2)

for bbox, color, label in [
    (result.water_bbox,   (255, 180, 0),  "water"),
    (result.food_bbox,    (0, 200, 80),   "food"),
    (result.bedding_bbox, (0, 0, 255),    "bedding"),
]:
    if bbox is not None:
        cv2.rectangle(viz,
            (int(bbox.x1), int(bbox.y1)),
            (int(bbox.x2), int(bbox.y2)),
            color, 2)
        cv2.putText(viz, label,
            (int(bbox.x1)+4, int(bbox.y1)+16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imwrite("debug_bbox_alignment.jpg", viz)
print(f"Saved → debug_bbox_alignment.jpg")
print(f"Water bbox : {result.water_bbox}")
print(f"Food bbox  : {result.food_bbox}")
print(f"Bedding bbox: {result.bedding_bbox}")