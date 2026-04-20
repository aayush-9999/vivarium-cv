# scripts/train.py
from ultralytics import YOLO
import multiprocessing
import os


def main():
    BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "dataset", "vivarium.yaml")

    print(f"\n📂 Using dataset: {DATA_PATH}\n")

    # Start from pretrained YOLOv8n — fine-tune on our 3 classes
    model = YOLO("yolov8n.pt")

    model.train(
        data=DATA_PATH,
        epochs=100,
        imgsz=640,
        batch=16,

        device=0,   # change to 0 for GPU
        workers=0,

        cos_lr=True,
        patience=20,

        # Augmentation — keep HSV augmentation for robustness under lab lighting
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        fliplr=0.5,
        mosaic=0.8,

        # Class weights — mouse is harder to detect (small, moves); boost it
        # Uncomment and adjust after first training run based on per-class metrics
        # cls=1.5,   # increase classification loss weight

        project=os.path.join(BASE_DIR, "runs", "detect"),
        name="vivarium_v1",
        exist_ok=True,
        verbose=True,
    )

    print("\n✅ Training complete.")
    print(f"   Weights: {BASE_DIR}/runs/detect/vivarium_v1/weights/best.pt")
    print(f"   Update YOLO_WEIGHTS in your .env to point to this path.\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()