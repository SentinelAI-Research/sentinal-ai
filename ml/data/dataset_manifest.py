from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    version: str
    source: str
    license: str
    purpose: str
    local_path: str
    sha256: str


def save_manifest(
    manifest: DatasetManifest,
    output_path: str,
) -> None:
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            asdict(manifest),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_manifest(
    input_path: str,
) -> DatasetManifest:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {input_path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    required_fields = {
        "name",
        "version",
        "source",
        "license",
        "purpose",
        "local_path",
        "sha256",
    }

    missing = required_fields - set(data)

    if missing:
        raise ValueError("Manifest is missing fields: " + ", ".join(sorted(missing)))

    return DatasetManifest(
        name=data["name"],
        version=data["version"],
        source=data["source"],
        license=data["license"],
        purpose=data["purpose"],
        local_path=data["local_path"],
        sha256=data["sha256"],
    )
