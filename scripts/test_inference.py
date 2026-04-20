from ultralytics import YOLO
import cv2

def main():
# Load model
    from ultralytics import YOLO

    model = YOLO(r"E:\AI\vivarium-project\vivarium-cv\runs\detect\vivarium_v1\weights\best.pt")


    # Load image
    img_path = r"dataset\augmented\images\image (3)_aug0037_wfull_fok.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Image not found: {img_path}")
        return

    # Inference
    results = model.predict(
        source=img,
        imgsz=640,
        conf=0.45,
        device="cpu"
    )

    # Debug: print class names
    print("Detected classes:", results[0].names)

    # Draw boxes
    annotated = results[0].plot()

    # Save output
    output_path = "output.jpg"
    cv2.imwrite(output_path, annotated)

    print(f"✅ Output saved at: {output_path}")

if __name__ == "__main__":
    main()
