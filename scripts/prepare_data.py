
from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets" / "raw"
OUTPUT_DIR = ROOT / "datasets" / "fruit_cls"

CLASSES = ["peach", "pomelo"]
SPLITS = ["train", "val", "test"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEED = 42

random.seed(SEED)


def get_valid_images(folder):
    images = []
    seen = set()

    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue

        try:
            with Image.open(path) as img:
                img.verify()

            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            if digest in seen:
                print(f"Anh trung lap: {path.name}")
                continue

            seen.add(digest)
            images.append(path)

        except Exception:
            print(f"Anh khong hop le: {path.name}")

    return images


def split_images(images):
    images = images.copy()
    random.shuffle(images)

    total = len(images)
    n_train = int(total * 0.7)
    n_val = int(total * 0.2)

    return {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }


def prepare_dataset():
    all_images = {}
    summary = defaultdict(dict)

    for class_name in CLASSES:
        source = RAW_DIR / class_name

        if not source.is_dir():
            raise FileNotFoundError(f"Khong tim thay: {source}")

        images = get_valid_images(source)

        if len(images) < 10:
            raise ValueError(
                f"{class_name} chi co {len(images)} anh hop le. "
                "Can it nhat 10 anh de chia du lieu."
            )

        all_images[class_name] = split_images(images)

    for split in SPLITS:
        for class_name in CLASSES:
            destination = OUTPUT_DIR / split / class_name
            destination.mkdir(parents=True, exist_ok=True)

            if any(destination.iterdir()):
                raise RuntimeError(
                    f"Thu muc dich khong rong: {destination}"
                )

    for class_name, partitions in all_images.items():
        for split, images in partitions.items():
            destination = OUTPUT_DIR / split / class_name

            for image in images:
                shutil.copy2(image, destination / image.name)

            summary[class_name][split] = len(images)

    print("\nKET QUA CHIA DATASET")
    print("-" * 45)

    for class_name in CLASSES:
        print(f"\n{class_name.upper()}")

        for split in SPLITS:
            print(f"  {split}: {summary[class_name][split]} anh")

    print("\nHoan thanh!")


if __name__ == "__main__":
    prepare_dataset()
