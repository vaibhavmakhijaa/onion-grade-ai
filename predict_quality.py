from pathlib import Path

import cv2
import numpy as np
import joblib
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

CLASSIFIER_PATH = (
    BASE_DIR
    / "models"
    / "onion_classifier.pkl"
)

IMAGE_PATH = (
    BASE_DIR
    / "raw_photos"
    / "overlapping.jpg"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading YOLO26...")
detector = YOLO(str(YOLO_MODEL_PATH))

print("Loading quality classifier...")
classifier = joblib.load(CLASSIFIER_PATH)


# ============================================================
# FEATURE EXTRACTION
# IMPORTANT:
# This MUST match train_classifier.py
# ============================================================

def extract_features(image):

    if image is None or image.size == 0:
        return None

    # Resize image
    image = cv2.resize(image, (128, 128))

    # --------------------------------------------------------
    # Focus on central part of onion
    # Reduces effect of:
    # - roots
    # - stems
    # - neighboring onions
    # - background
    # --------------------------------------------------------

    h, w = image.shape[:2]

    x1 = int(w * 0.15)
    x2 = int(w * 0.85)

    y1 = int(h * 0.15)
    y2 = int(h * 0.85)

    image = image[y1:y2, x1:x2]

    # Resize again
    image = cv2.resize(image, (128, 128))

    # Convert BGR → HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --------------------------------------------------------
    # COLOR FEATURES
    # --------------------------------------------------------

    hue = np.mean(hsv[:, :, 0])

    saturation = np.mean(hsv[:, :, 1])

    value = np.mean(hsv[:, :, 2])

    # --------------------------------------------------------
    # DARK PIXEL RATIO
    # --------------------------------------------------------

    dark_pixels = np.sum(
        hsv[:, :, 2] < 50
    )

    dark_pixel_ratio = (
        dark_pixels / (128 * 128)
    )

    # --------------------------------------------------------
    # GREEN PIXEL RATIO
    # Used mainly for sprouting detection
    # --------------------------------------------------------

    green_mask = (
        (hsv[:, :, 0] > 35)
        &
        (hsv[:, :, 0] < 90)
        &
        (hsv[:, :, 1] > 60)
    )

    green_pixel_ratio = (
        np.sum(green_mask)
        / (128 * 128)
    )

    # --------------------------------------------------------
    # EXACTLY 5 FEATURES
    # --------------------------------------------------------

    return [
        hue,
        saturation,
        value,
        dark_pixel_ratio,
        green_pixel_ratio
    ]


# ============================================================
# CHECK IMAGE
# ============================================================

print("\nChecking input image...")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"\nImage not found:\n{IMAGE_PATH}\n"
        "\nCheck the filename inside raw_photos."
    )

print("Image:", IMAGE_PATH)


# ============================================================
# LOAD IMAGE
# ============================================================

original = cv2.imread(str(IMAGE_PATH))

if original is None:
    raise ValueError(
        "Could not read the input image."
    )


# ============================================================
# YOLO DETECTION
# ============================================================

print("\nDetecting onions...\n")

results = detector.predict(
    source=str(IMAGE_PATH),
    conf=0.50
)

result = results[0]

onion_count = len(result.boxes)

print(
    f"Detected onions: {onion_count}"
)


# ============================================================
# CLASS COUNTS
# ============================================================

counts = {
    "healthy": 0,
    "damaged": 0,
    "rotten": 0,
    "sprouted": 0
}


# ============================================================
# INDIVIDUAL PREDICTIONS
# ============================================================

print("\nIndividual predictions:")
print("--------------------------------")


for i, box in enumerate(result.boxes):

    # Get bounding box
    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0]
    )

    # --------------------------------------------------------
    # Add small margin around detection
    # --------------------------------------------------------

    width = x2 - x1
    height = y2 - y1

    margin_x = int(width * 0.05)
    margin_y = int(height * 0.05)

    x1_crop = max(
        0,
        x1 + margin_x
    )

    y1_crop = max(
        0,
        y1 + margin_y
    )

    x2_crop = min(
        original.shape[1],
        x2 - margin_x
    )

    y2_crop = min(
        original.shape[0],
        y2 - margin_y
    )

    # --------------------------------------------------------
    # Crop onion
    # --------------------------------------------------------

    crop = original[
        y1_crop:y2_crop,
        x1_crop:x2_crop
    ]

    if crop.size == 0:
        print(
            f"Onion {i + 1}: Could not crop"
        )
        continue

    # --------------------------------------------------------
    # Extract EXACT same 5 features
    # used during training
    # --------------------------------------------------------

    features = extract_features(crop)

    if features is None:
        print(
            f"Onion {i + 1}: Feature extraction failed"
        )
        continue

    # --------------------------------------------------------
    # Predict quality
    # --------------------------------------------------------

    prediction = classifier.predict(
        [features]
    )[0]

    # --------------------------------------------------------
    # Prediction probability
    # --------------------------------------------------------

    probabilities = classifier.predict_proba(
        [features]
    )[0]

    confidence = max(probabilities)

    # --------------------------------------------------------
    # Update count
    # --------------------------------------------------------

    counts[prediction] += 1

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"Onion {i + 1}: "
        f"{prediction.upper()} "
        f"({confidence * 100:.1f}%)"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n================================")
print("QUALITY SUMMARY")
print("================================")

print(
    f"Total onions: {onion_count}"
)

for category, count in counts.items():

    percentage = (
        count / onion_count * 100
        if onion_count > 0
        else 0
    )

    print(
        f"{category.capitalize():10s}: "
        f"{count:2d} "
        f"({percentage:.1f}%)"
    )


# ============================================================
# APPROXIMATE GRADE
# ============================================================

if onion_count > 0:

    healthy_percentage = (
        counts["healthy"]
        / onion_count
        * 100
    )

    if (
        healthy_percentage >= 80
        and counts["rotten"] == 0
        and counts["sprouted"] == 0
    ):
        grade = "GRADE A"
    else:
        grade = "URS / LOWER GRADE"

    print("\n================================")
    print("APPROXIMATE GRADING")
    print("================================")

    print(
        f"Healthy / Grade A %: "
        f"{healthy_percentage:.1f}%"
    )

    print(
        f"Prototype Result: {grade}"
    )


print("\nAnalysis complete! 🧅")