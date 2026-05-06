# scripts/train.py
from ultralytics import YOLO
import multiprocessing
import os
from yolox.core import Trainer
from yolox.exp import get_exp



def main():
    BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "dataset", "vivarium.yaml")

    print(f"\n📂 Using dataset: {DATA_PATH}\n")


    exp = get_exp("exps/vivarium_yolox_tiny.py")
    exp.merge(["max_epoch=100", "basic_lr_per_img=0.01/64"])
    trainer = Trainer(exp, args)
    trainer.train()

    print("\n✅ Training complete.")
    print(f"   Weights: {BASE_DIR}/runs/detect/vivarium_v1/weights/best.pt")

    
    print(f"   Update YOLO_WEIGHTS in your .env to point to this path.\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()