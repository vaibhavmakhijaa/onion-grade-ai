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

data_path = BASE_DIR / "yolo_dataset" / "data.yaml"

model = YOLO(str(model_path))

metrics = model.val(
    data=str(data_path),
    split="test",
    imgsz=512
)

print("Test evaluation complete!")
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)