
from pathlib import Path
from collections import defaultdict
import hashlib

BASE_DIR = Path(r"D:\BrainTumourDetection\braintumour")
TRAIN_DIR = BASE_DIR / "Training"
TEST_DIR = BASE_DIR / "Testing"


def get_hash(path):
    """Calculate a hash of the original file bytes."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def collect_images(root):
    """Collect image paths grouped by their file contents."""
    groups = defaultdict(list)

    for path in root.glob("*/*"):
        if path.is_file():
            try:
                groups[get_hash(path)].append(path)
            except OSError as error:
                print(f"Could not read {path}: {error}")

    return groups


print("Checking training images...")
train_groups = collect_images(TRAIN_DIR)

print("Checking testing images...")
test_groups = collect_images(TEST_DIR)

train_hashes = set(train_groups)
test_hashes = set(test_groups)

common_hashes = train_hashes & test_hashes

print("\nDATASET DUPLICATE REPORT")
print("=" * 40)
print("Unique training image contents:", len(train_hashes))
print("Unique testing image contents:", len(test_hashes))
print("Exact duplicate contents across splits:", len(common_hashes))

if common_hashes:
    print("\nExamples of matching files:")
    for digest in list(common_hashes)[:10]:
        print("\nTraining:")
        for path in train_groups[digest]:
            print(" ", path)

        print("Testing:")
        for path in test_groups[digest]:
            print(" ", path)

print("\nCheck completed.")
