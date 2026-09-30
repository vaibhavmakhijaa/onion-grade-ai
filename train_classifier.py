from pathlib import Path
import cv2
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "healthy",
    "damaged",
    "rotten",
    "sprouted"
]


# ============================================================
# FEATURE EXTRACTION
# IMPORTANT:
# This exact function is also used by the Streamlit app.
# DO NOT change one without changing the other.
# ============================================================

def extract_features(image):

    if image is None or image.size == 0:
        return None

    # Resize
    image = cv2.resize(image, (128, 128))

    # Convert BGR -> HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # --------------------------------------------------------
    # FEATURE 1 - HUE
    # --------------------------------------------------------

    hue = np.mean(hsv[:, :, 0])

    # --------------------------------------------------------
    # FEATURE 2 - SATURATION
    # --------------------------------------------------------

    saturation = np.mean(hsv[:, :, 1])

    # --------------------------------------------------------
    # FEATURE 3 - VALUE / BRIGHTNESS
    # --------------------------------------------------------

    value = np.mean(hsv[:, :, 2])

    # --------------------------------------------------------
    # FEATURE 4 - DARK PIXEL RATIO
    # --------------------------------------------------------

    dark_pixels = np.sum(
        hsv[:, :, 2] < 80
    )

    dark_pixel_ratio = (
        dark_pixels / (128 * 128)
    )

    # --------------------------------------------------------
    # FEATURE 5 - GREEN PIXEL RATIO
    # --------------------------------------------------------

    green_mask = (
        (hsv[:, :, 0] > 35) &
        (hsv[:, :, 0] < 90) &
        (hsv[:, :, 1] > 60)
    )

    green_pixel_ratio = (
        np.sum(green_mask) / (128 * 128)
    )

    # --------------------------------------------------------
    # RETURN EXACTLY 5 FEATURES
    # --------------------------------------------------------

    return [
        hue,
        saturation,
        value,
        dark_pixel_ratio,
        green_pixel_ratio
    ]


# ============================================================
# LOAD DATASET
# ============================================================

X = []
y = []

print("\n========================================")
print("     ONION QUALITY CLASSIFIER")
print("========================================")
print("\nLoading dataset...\n")


for class_name in CLASSES:

    folder = DATA_DIR / class_name

    images = (
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.jpeg"))
        + list(folder.glob("*.png"))
        + list(folder.glob("*.webp"))
    )

    print(
        f"{class_name:10s}: {len(images)} images"
    )

    for image_path in images:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print(
                f"WARNING: Could not read {image_path.name}"
            )
            continue

        features = extract_features(
            image
        )

        if features is not None:

            X.append(features)
            y.append(class_name)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(X)
y = np.array(y)


print("\n----------------------------------------")
print("Total images:", len(X))
print("Feature shape:", X.shape)
print("----------------------------------------")


# ============================================================
# CHECK DATASET
# ============================================================

if len(X) == 0:

    raise RuntimeError(
        "No training images found. "
        "Check the data/healthy, data/damaged, "
        "data/rotten and data/sprouted folders."
    )


print("\nClass distribution:")

for class_name in CLASSES:

    count = np.sum(
        y == class_name
    )

    print(
        f"{class_name:10s}: {count}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n----------------------------------------")
print("Training images:", len(X_train))
print("Testing images :", len(X_test))
print("----------------------------------------")


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...\n")


model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# EVALUATION
# ============================================================

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n========================================")
print("       CLASSIFIER RESULTS")
print("========================================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR
    / "onion_classifier.pkl"
)


joblib.dump(
    model,
    model_path
)


print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(
    model_path
)

print("\nTraining complete! 🧅")