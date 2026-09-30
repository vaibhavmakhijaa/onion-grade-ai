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


# ============================================================
# IMAGES TO TEST
# ============================================================

IMAGES = [
    "overlapping.jpg",
    "tray_photo22.webp"
]


# ============================================================
# FEATURE EXTRACTION
# MUST MATCH CURRENT TRAINING
# ============================================================

def extract_features(image):

    if image is None or image.size == 0:
        return None

    image = cv2.resize(
        image,
        (128, 128)
    )

    # Central region
    h, w = image.shape[:2]

    x1 = int(w * 0.15)
    x2 = int(w * 0.85)

    y1 = int(h * 0.15)
    y2 = int(h * 0.85)

    image = image[
        y1:y2,
        x1:x2
    ]

    image = cv2.resize(
        image,
        (128, 128)
    )

    # BGR -> HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    hue = np.mean(
        hsv[:, :, 0]
    )

    saturation = np.mean(
        hsv[:, :, 1]
    )

    value = np.mean(
        hsv[:, :, 2]
    )

    dark_pixels = np.sum(
        hsv[:, :, 2] < 50
    )

    dark_pixel_ratio = (
        dark_pixels / (128 * 128)
    )

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

    return [
        hue,
        saturation,
        value,
        dark_pixel_ratio,
        green_pixel_ratio
    ]


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading YOLO26...")
detector = YOLO(
    str(YOLO_MODEL_PATH)
)

print("Loading current classifier...")
classifier = joblib.load(
    CLASSIFIER_PATH
)


# ============================================================
# TEST EACH IMAGE
# ============================================================

for image_name in IMAGES:

    image_path = (
        BASE_DIR
        / "raw_photos"
        / image_name
    )

    print("\n")
    print("=" * 60)
    print(f"TESTING: {image_name}")
    print("=" * 60)

    if not image_path.exists():

        print(
            f"SKIPPED - file not found:\n{image_path}"
        )

        continue

    original = cv2.imread(
        str(image_path)
    )

    if original is None:

        print(
            "ERROR - image could not be read."
        )

        continue

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    results = detector.predict(
        source=str(image_path),
        conf=0.50
    )

    result = results[0]

    onion_count = len(
        result.boxes
    )

    print(
        f"\nDetected onions: {onion_count}"
    )

    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    counts = {
        "healthy": 0,
        "damaged": 0,
        "rotten": 0,
        "sprouted": 0
    }

    # --------------------------------------------------------
    # CLASSIFY
    # --------------------------------------------------------

    for i, box in enumerate(
        result.boxes
    ):

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        width = x2 - x1
        height = y2 - y1

        margin_x = int(
            width * 0.05
        )

        margin_y = int(
            height * 0.05
        )

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

        crop = original[
            y1_crop:y2_crop,
            x1_crop:x2_crop
        ]

        if crop.size == 0:
            continue

        features = extract_features(
            crop
        )

        prediction = classifier.predict(
            [features]
        )[0]

        probabilities = classifier.predict_proba(
            [features]
        )[0]

        confidence = max(
            probabilities
        )

        counts[prediction] += 1

        print(
            f"Onion {i + 1:02d}: "
            f"{prediction.upper():8s} "
            f"{confidence * 100:5.1f}%"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\nSUMMARY")

    for category, count in counts.items():

        percentage = (
            count / onion_count * 100
            if onion_count > 0
            else 0
        )

        print(
            f"{category.capitalize():10s}: "
            f"{count:2d} "
            f"({percentage:5.1f}%)"
        )


print("\n")
print("=" * 60)
print("DIAGNOSTIC TEST COMPLETE")
print("=" * 60)