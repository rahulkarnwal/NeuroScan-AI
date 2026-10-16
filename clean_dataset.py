
from pathlib import Path
from collections import defaultdict
import hashlib
import shutil

# Original dataset: never modify this folder
SOURCE_DIR = Path(
    r"D:\BrainTumourDetection\braintumour"
)

# New destination for the cleaned dataset
DEST_DIR = Path(
    r"D:\BrainTumourDetection\braintumour_cleaned"
)


def file_hash(path):
    """Calculate SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def clean_split(split_name):
    source_split = SOURCE_DIR / split_name
    dest_split = DEST_DIR / split_name

    if not source_split.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {source_split}"
        )

    if dest_split.exists():
        raise FileExistsError(
            f"{dest_split} already exists.\n"
            "Choose a new destination or inspect the existing "
            "folder before running again."
        )

    total_files = 0
    kept_files = 0
    removed_duplicates = 0

    print(f"\nProcessing {split_name}...")

    for class_dir in sorted(source_split.iterdir()):
        if not class_dir.is_dir():
            continue

        destination_class = dest_split / class_dir.name
        destination_class.mkdir(
            parents=True,
            exist_ok=True
        )

        # Hashes are tracked separately for each class.
        seen_hashes = set()

        for source_file in sorted(class_dir.iterdir()):
            if not source_file.is_file():
                continue

            total_files += 1

            try:
                digest = file_hash(source_file)
            except OSError as error:
                raise RuntimeError(
                    f"Cannot read {source_file}: {error}"
                ) from error

            if digest in seen_hashes:
                removed_duplicates += 1
                continue

            seen_hashes.add(digest)

            destination_file = (
                destination_class / source_file.name
            )

            shutil.copy2(source_file, destination_file)
            kept_files += 1

        print(
            f"  {class_dir.name}: "
            f"{len(seen_hashes)} unique files"
        )

    print(f"\n{split_name} summary:")
    print(f"  Original files: {total_files}")
    print(f"  Files retained: {kept_files}")
    print(f"  Duplicate copies skipped: {removed_duplicates}")

    return total_files, kept_files, removed_duplicates


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_DIR}"
        )

    if DEST_DIR.exists():
        raise FileExistsError(
            f"Destination already exists: {DEST_DIR}\n"
            "No files were changed. Inspect it or select "
            "a different destination."
        )

    print("Original dataset:", SOURCE_DIR)
    print("Cleaned dataset:", DEST_DIR)
    print("\nThe original dataset will not be modified.")

    totals = {}

    for split in ["Training", "Testing"]:
        totals[split] = clean_split(split)

    print("\n" + "=" * 45)
    print("CLEANING COMPLETED")
    print("=" * 45)

    for split, (original, kept, removed) in totals.items():
        print(
            f"{split}: {original} original, "
            f"{kept} retained, {removed} duplicates skipped"
        )

    print("\nCleaned dataset location:")
    print(DEST_DIR)


if __name__ == "__main__":
    main()
