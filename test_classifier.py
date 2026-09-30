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
# ============================================================

def extract_features(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    # Resize image
    image = cv2.resize(image, (128, 128))

    # --------------------------------------------------------
    # Focus more on the central part of the onion
    # This reduces the effect of:
    # - roots
    # - stems
    # - neighboring onions
    # - dark background
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
    # BASIC COLOR FEATURES
    # --------------------------------------------------------

    hue = np.mean(hsv[:, :, 0])

    saturation = np.mean(hsv[:, :, 1])

    value = np.mean(hsv[:, :, 2])

    # --------------------------------------------------------
    # DARK PIXEL RATIO
    # --------------------------------------------------------
    # Lower threshold than before so normal onion shading
    # is less likely to be treated as damage.
    # --------------------------------------------------------

    dark_pixels = np.sum(hsv[:, :, 2] < 50)

    dark_pixel_ratio = dark_pixels / (128 * 128)

    # --------------------------------------------------------
    # GREEN PIXEL RATIO
    # Useful for detecting sprouting.
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
    # FINAL FEATURE VECTOR
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

print("\n======================================")
print("ONION QUALITY CLASSIFIER TRAINING")
print("======================================\n")

print("Loading dataset...\n")


for class_name in CLASSES:

    folder = DATA_DIR / class_name

    images = (
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.jpeg"))
        + list(folder.glob("*.png"))
        + list(folder.glob("*.webp"))
    )

    print(f"{class_name}: {len(images)} images")

    for image_path in images:

        features = extract_features(image_path)

        if features is not None:

            X.append(features)
            y.append(class_name)


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(X)
y = np.array(y)

print("\n--------------------------------------")
print("Dataset loaded")
print("--------------------------------------")

print("Total images:", len(X))
print("Feature shape:", X.shape)


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


print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest...\n")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)


# ============================================================
# EVALUATION
# ============================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\n======================================")
print("CLASSIFIER RESULTS")
print("======================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
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

model_path = MODEL_DIR / "onion_classifier.pkl"

joblib.dump(
    model,
    model_path
)


print("\n======================================")
print("MODEL SAVED")
print("======================================")

print(model_path)

print("\nTraining complete! 🧅")