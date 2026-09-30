from pathlib import Path
import cv2
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

YOLO_MODEL_PATH = (
    BASE_DIR.parent
    / "runs"
    / "detect"
    / "onion_detector_v3"
    / "weights"
    / "best.pt"
)

IMAGE_PATH = (
    BASE_DIR
    / "raw_photos"
    / "overlapping.jpg"
)

HEALTHY_DIR = BASE_DIR / "data" / "healthy"


# ============================================================
# CREATE HEALTHY FOLDER
# ============================================================

HEALTHY_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO26...")

model = YOLO(
    str(YOLO_MODEL_PATH)
)


# ============================================================
# CHECK IMAGE
# ============================================================

if not IMAGE_PATH.exists():

    raise FileNotFoundError(
        f"\nImage not found:\n{IMAGE_PATH}\n"
        "\nChange IMAGE_PATH to the actual filename."
    )


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:

    raise ValueError(
        "Could not read the image."
    )


# ============================================================
# DETECT ONIONS
# ============================================================

print("\nDetecting onions...")

results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.50
)

result = results[0]

print(
    f"\nDetected onions: {len(result.boxes)}"
)


# ============================================================
# SAVE CROPS
# ============================================================

saved = 0

print("\nSaving healthy onion crops...")
print("--------------------------------")


for i, box in enumerate(result.boxes):

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0]
    )

    width = x2 - x1
    height = y2 - y1

    # Small margin
    margin_x = int(width * 0.05)
    margin_y = int(height * 0.05)

    x1 = max(
        0,
        x1 + margin_x
    )

    y1 = max(
        0,
        y1 + margin_y
    )

    x2 = min(
        image.shape[1],
        x2 - margin_x
    )

    y2 = min(
        image.shape[0],
        y2 - margin_y
    )

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:
        continue

    filename = (
        HEALTHY_DIR
        / f"batch_healthy_{saved + 1}.jpg"
    )

    cv2.imwrite(
        str(filename),
        crop
    )

    saved += 1

    print(
        f"Saved: {filename.name}"
    )


# ============================================================
# RESULT
# ============================================================

print("\n======================================")
print("BATCH CROPPING COMPLETE")
print("======================================")

print(
    f"Healthy crops saved: {saved}"
)

print(
    f"Healthy dataset location:\n{HEALTHY_DIR}"
)

print("\nThese crops will now be used as")
print("real-world healthy training examples. 🧅")