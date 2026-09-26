from ml.data.dataset_manifest import (
    DatasetManifest,
    load_manifest,
    save_manifest,
)


def test_dataset_manifest_round_trip(tmp_path):
    manifest = DatasetManifest(
        name="sentinelai-controlled-source",
        version="1.0",
        source="local-controlled-benchmark",
        license="project-controlled",
        purpose="Runtime provenance testing",
        local_path="data/sample/confidential_document.txt",
        sha256="a" * 64,
    )

    output_path = tmp_path / "manifest.json"

    save_manifest(
        manifest,
        str(output_path),
    )

    loaded = load_manifest(str(output_path))

    assert loaded == manifest
