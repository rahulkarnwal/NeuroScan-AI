
from pathlib import Path
from collections import defaultdict
import hashlib

BASE_DIR = Path(
    r"D:\BrainTumourDetection\braintumour_cleaned"
)


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


for split in ["Training", "Testing"]:
    root = BASE_DIR / split
    total_files = 0
    duplicates = 0

    print(f"\n{split.upper()}")

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        seen = set()
        class_total = 0

        for path in class_dir.iterdir():
            if not path.is_file():
                continue

            class_total += 1
            total_files += 1
            digest = file_hash(path)

            if digest in seen:
                duplicates += 1
            else:
                seen.add(digest)

        print(f"{class_dir.name}: {class_total} images")

    print("Total images:", total_files)
    print("Remaining exact duplicates:", duplicates)

print("\nVerification completed.")
