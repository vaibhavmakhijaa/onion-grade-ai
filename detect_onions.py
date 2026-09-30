import cv2
import os


def detect_onions(image_path, min_area=2000):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(
            f"Could not read image at '{image_path}'."
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    onions = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)

        onions.append({
            "crop": img[y:y+h, x:x+w],
            "bbox": (x, y, w, h)
        })

    return img, onions


if __name__ == "__main__":
    photo_path = "raw_photos/tray_photo88.jpeg"

    img, onions = detect_onions(photo_path)

    print(f"Found {len(onions)} onions.")

    os.makedirs("detected_crops", exist_ok=True)

    for i, o in enumerate(onions):
        cv2.imwrite(
            f"detected_crops/onion_{i}.jpg",
            o["crop"]
        )

    preview = img.copy()

    for o in onions:
        x, y, w, h = o["bbox"]

        cv2.rectangle(
            preview,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

    cv2.imwrite("detection_preview.jpg", preview)

    print("Open detection_preview.jpg to check.")