import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.data.dataset_manifest import DatasetManifest

DATASET_NAME = "AMI Meeting Corpus"
DATASET_VERSION = "knkarthick/AMI"
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "ami"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "ami_manifest.json"


def main() -> None:
    print("SentinelAI - Creating AMI dataset manifest")
    print(f"Dataset directory: {DATASET_DIR}")

    manifest = DatasetManifest.from_directory(
        dataset_name=DATASET_NAME,
        dataset_version=DATASET_VERSION,
        directory=DATASET_DIR,
    )

    manifest.save(MANIFEST_PATH)

    print("\nManifest created successfully.")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Files recorded: {len(manifest.files)}")

    print("\nFiles:")

    for file_info in manifest.files:
        print(
            f"  {file_info['path']} | "
            f"{file_info['size_bytes']} bytes | "
            f"{file_info['sha256']}"
        )


if __name__ == "__main__":
    main()
