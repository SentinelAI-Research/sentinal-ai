from __future__ import annotations

import hashlib
import sys
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DOWNLOAD_URL = (
    "https://www.cs.cmu.edu/~enron/"
    "enron_mail_20150507.tar.gz"
)

RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "enron"
ARCHIVE_PATH = RAW_DIRECTORY / "enron_mail_20150507.tar.gz"


def download_archive() -> None:
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists():
        print(f"Archive already exists: {ARCHIVE_PATH}")
        return

    print("Downloading Enron dataset from the official CMU source...")
    print(DOWNLOAD_URL)

    request = Request(
        DOWNLOAD_URL,
        headers={
            "User-Agent": "SentinelAI-Research/1.0",
        },
    )

    with urlopen(request, timeout=120) as response:
        total_size = response.headers.get("Content-Length")

        if total_size is not None:
            total_size_int = int(total_size)
            print(
                f"Expected download size: "
                f"{total_size_int / (1024 * 1024):.2f} MB"
            )
        else:
            total_size_int = None
            print("Download size was not provided by the server.")

        downloaded = 0

        with ARCHIVE_PATH.open("wb") as output_file:
            while True:
                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                output_file.write(chunk)
                downloaded += len(chunk)

                if total_size_int:
                    percentage = downloaded / total_size_int * 100
                    print(
                        f"\rDownloaded: "
                        f"{downloaded / (1024 * 1024):.2f} MB "
                        f"({percentage:.1f}%)",
                        end="",
                    )
                else:
                    print(
                        f"\rDownloaded: "
                        f"{downloaded / (1024 * 1024):.2f} MB",
                        end="",
                    )

    print()
    print(f"Archive saved to: {ARCHIVE_PATH}")


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def verify_archive() -> None:
    print("Verifying downloaded archive...")

    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"Expected archive does not exist: {ARCHIVE_PATH}"
        )

    try:
        with tarfile.open(ARCHIVE_PATH, mode="r:gz") as archive:
            members = archive.getmembers()

        if not members:
            raise ValueError("The Enron archive is empty.")

        print(f"Archive contains {len(members):,} entries.")
        print("Archive integrity check: PASSED")

    except tarfile.TarError as exc:
        raise RuntimeError(
            "The downloaded file is not a valid gzip-compressed tar archive."
        ) from exc

    sha256 = calculate_sha256(ARCHIVE_PATH)

    print(f"Archive SHA-256: {sha256}")


def main() -> None:
    download_archive()
    verify_archive()

    print()
    print("Enron dataset acquisition completed successfully.")
    print()
    print("Next step:")
    print("  1. Inspect the archive structure.")
    print("  2. Extract it only after the acquisition check passes.")


if __name__ == "__main__":
    main()