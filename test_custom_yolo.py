from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

model_path = (
    BASE_DIR.parent
    / "runs"
    / "detect"
    / "onion_detector_v3"
    / "weights"
    / "best.pt"
)

image_path = BASE_DIR / "raw_photos" / "tray_photo.jpeg"

print("Model:", model_path)
print("Image:", image_path)

model = YOLO(str(model_path))

results = model.predict(
    source=str(image_path),
    conf=0.50,
    save=True
)

print("Custom YOLO26 detection complete!")