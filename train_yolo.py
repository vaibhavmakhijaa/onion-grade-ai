from pathlib import Path
from ultralytics import YOLO

# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Version 3 dataset
DATASET = BASE_DIR / "yolo_dataset_v3" / "data.yaml"

print("Dataset:", DATASET)

# Load YOLO26 Nano
model = YOLO("yolo26n.pt")

# Train
results = model.train(
    data=str(DATASET),
    epochs=50,
    imgsz=512,
    batch=4,
    name="onion_detector_v3"
)

print("\nTraining complete!")