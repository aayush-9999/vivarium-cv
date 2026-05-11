# pipeline/storage/image_store/local.py
import os
import cv2
from datetime import datetime, timezone

def save_flagged_frame(
    frame,
    cage_id: str,
    output_dir: str = "flagged_frames",
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(output_dir, f"{cage_id}_{ts}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path