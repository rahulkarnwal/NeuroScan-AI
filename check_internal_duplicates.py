
from pathlib import Path
from collections import defaultdict
import hashlib

BASE_DIR = Path(r"D:\BrainTumourDetection\braintumour")


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


for split in ["Training", "Testing"]:
    root = BASE_DIR / split
    groups = defaultdict(list)

    for path in root.glob("*/*"):
        if path.is_file():
            try:
                groups[file_hash(path)].append(path)
            except OSError as error:
                print(f"Could not read {path}: {error}")

    duplicate_groups = [
        paths for paths in groups.values()
        if len(paths) > 1
    ]

    cross_class_groups = [
        paths for paths in duplicate_groups
        if len({path.parent.name for path in paths}) > 1
    ]

    print(f"\n{split.upper()} REPORT")
    print("=" * 45)
    print("Duplicate content groups:", len(duplicate_groups))
    print(
        "Files in duplicate groups:",
        sum(len(paths) for paths in duplicate_groups)
    )
    print("Duplicates across different class folders:", len(cross_class_groups))

    for paths in duplicate_groups[:10]:
        print("\nMatching files:")
        for path in paths:
            print(" ", path.relative_to(root))

    if cross_class_groups:
        print("\nWARNING: Identical files appear under different labels.")
