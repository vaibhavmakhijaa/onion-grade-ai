from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR / "yolo_dataset_v3"

for split in ["train", "valid", "test"]:
    labels_dir = ROOT / split / "labels"

    bad = []

    for file in labels_dir.glob("*.txt"):
        for line_no, line in enumerate(file.read_text().splitlines(), 1):
            if line.strip() and len(line.split()) != 5:
                bad.append((file.name, line_no, len(line.split())))

    print(f"\n{split}:")
    print("Bad annotation lines:", len(bad))

    for item in bad[:10]:
        print(item)