from pathlib import Path

from ml.data.hashing import sha256_file, sha256_text


def test_sha256_text_is_deterministic():
    first = sha256_text("SentinelAI")
    second = sha256_text("SentinelAI")

    assert first == second
    assert len(first) == 64


def test_sha256_file_is_deterministic(tmp_path: Path):
    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "SentinelAI reproducibility test",
        encoding="utf-8",
    )

    first = sha256_file(str(file_path))
    second = sha256_file(str(file_path))

    assert first == second
    assert len(first) == 64
