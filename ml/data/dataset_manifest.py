from datetime import datetime, timezone
from pathlib import Path

import json

from ml.data.hashing import sha256_file


class DatasetManifest:
    """
    Dataset metadata and reproducibility manifest.

    Supports the original SentinelAI single-source manifest
    contract as well as multi-file dataset manifests.
    """

    def __init__(
        self,
        name: str | None = None,
        version: str | None = None,
        source: str | None = None,
        license: str | None = None,
        purpose: str | None = None,
        local_path: str | None = None,
        sha256: str | None = None,
        *,
        dataset_name: str | None = None,
        dataset_version: str | None = None,
        files: list[dict] | None = None,
        created_at: str | None = None,
    ) -> None:
        resolved_name = dataset_name if dataset_name is not None else name

        resolved_version = dataset_version if dataset_version is not None else version

        if not resolved_name:
            raise ValueError("Dataset name must not be empty.")

        if not resolved_version:
            raise ValueError("Dataset version must not be empty.")

        self.name = resolved_name
        self.version = resolved_version

        self.source = source
        self.license = license
        self.purpose = purpose
        self.local_path = local_path
        self.sha256 = sha256

        self.files = files or []

        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_directory(
        cls,
        dataset_name: str,
        dataset_version: str,
        directory: str | Path,
    ) -> "DatasetManifest":
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Expected a directory: {directory}")

        files = []

        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue

            files.append(
                {
                    "path": path.relative_to(directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

        if not files:
            raise ValueError(f"No files found in dataset directory: {directory}")

        return cls(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            files=files,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "license": self.license,
            "purpose": self.purpose,
            "local_path": self.local_path,
            "sha256": self.sha256,
            "dataset_name": self.name,
            "dataset_version": self.version,
            "created_at": self.created_at,
            "files": self.files,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DatasetManifest):
            return NotImplemented

        return self.to_dict() == other.to_dict()

    def save(
        self,
        output_path: str | Path,
    ) -> None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.to_dict(),
                file,
                indent=2,
                ensure_ascii=False,
            )


def save_manifest(
    manifest: DatasetManifest,
    output_path: str | Path,
) -> None:
    """
    Save a DatasetManifest to JSON.
    """

    manifest.save(output_path)


def load_manifest(
    input_path: str | Path,
) -> DatasetManifest:
    """
    Load a DatasetManifest from JSON.

    Supports both the original SentinelAI manifest format
    and the newer multi-file format.
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {input_path}")

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return DatasetManifest(
        name=data.get(
            "name",
            data.get("dataset_name"),
        ),
        version=data.get(
            "version",
            data.get("dataset_version"),
        ),
        source=data.get("source"),
        license=data.get("license"),
        purpose=data.get("purpose"),
        local_path=data.get("local_path"),
        sha256=data.get("sha256"),
        files=data.get("files", []),
        created_at=data.get("created_at"),
    )
