
from pathlib import Path
from collections import defaultdict
import hashlib

BASE_DIR = Path(r"D:\BrainTumourDetection\braintumour")


def file_hash(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


for split in ["Training", "Testing"]:
    root = BASE_DIR / split
    groups = defaultdict(list)

    for path in root.glob("*/*"):
        if path.is_file():
            groups[file_hash(path)].append(path)

    print(f"\n{split.upper()}")
    print("-" * 50)

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_groups = [
            paths for paths in groups.values()
            if len(paths) > 1
            and all(p.parent == class_dir for p in paths)
        ]

        duplicate_files = sum(
            len(paths) - 1 for paths in class_groups
        )

        print(
            f"{class_dir.name:12} | "
            f"duplicate groups: {len(class_groups):3} | "
            f"extra copies: {duplicate_files:3}"
        )
