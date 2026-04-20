from ultralytics import YOLO
import multiprocessing
import os


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "dataset", "vivarium.yaml")

    print(f"\n📂 Using dataset: {DATA_PATH}\n")

    model = YOLO("yolov8n.pt")

    # model.train(
    #     data=DATA_PATH,
    #     epochs=100,
    #     imgsz=640,
    #     batch=16,

    #     device= 0,  # switch to 0 later
    #     workers=0,

    #     cos_lr=True,
    #     patience=20,

    #     hsv_h=0.015,
    #     hsv_s=0.4,
    #     hsv_v=0.3,
    #     fliplr=0.5,
    #     mosaic=0.8,

    #     project=os.path.join(BASE_DIR, "runs", "detect"),
    #     name="vivarium_v1",
    #     exist_ok=True,
    #     verbose=True,
    # )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()