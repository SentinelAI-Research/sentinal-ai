from __future__ import annotations

import re
import sys
import tarfile
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "enron"
ARCHIVE_PATH = RAW_DIRECTORY / "enron_mail_20150507.tar.gz"


EMAIL_HEADER_PATTERN = re.compile(
    rb"^(From|Date|Message-ID|Subject|To|Cc):",
    re.MULTILINE,
)


def inspect_archive() -> list[tarfile.TarInfo]:
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"Enron archive not found: {ARCHIVE_PATH}"
        )

    with tarfile.open(ARCHIVE_PATH, mode="r:gz") as archive:
        members = archive.getmembers()

    print("=" * 70)
    print("ENRON ARCHIVE INSPECTION")
    print("=" * 70)

    print(f"Archive: {ARCHIVE_PATH}")
    print(f"Total archive entries: {len(members):,}")

    directories = [
        member
        for member in members
        if member.isdir()
    ]

    files = [
        member
        for member in members
        if member.isfile()
    ]

    print(f"Directories: {len(directories):,}")
    print(f"Files: {len(files):,}")

    print()
    print("First 20 archive paths:")
    print("-" * 70)

    for member in members[:20]:
        print(member.name)

    return files


def inspect_archive_structure(
    files: list[tarfile.TarInfo],
) -> None:
    print()
    print("=" * 70)
    print("ARCHIVE FILE STRUCTURE")
    print("=" * 70)

    suffix_counts = Counter(
        Path(member.name).suffix.lower() or "<no extension>"
        for member in files
    )

    print("File extensions:")

    for suffix, count in suffix_counts.most_common():
        print(f"  {suffix}: {count:,}")

    print()
    print("Sample file paths:")

    for member in files[:20]:
        print(f"  {member.name}")


def identify_email_members(
    files: list[tarfile.TarInfo],
) -> list[tarfile.TarInfo]:
    """
    Identify raw email files directly inside the archive.

    No files are extracted to disk.
    """

    candidates: list[tarfile.TarInfo] = []

    with tarfile.open(ARCHIVE_PATH, mode="r:gz") as archive:
        for member in files:
            extracted = archive.extractfile(member)

            if extracted is None:
                continue

            sample = extracted.read(8192)

            if EMAIL_HEADER_PATTERN.search(sample):
                candidates.append(member)

            if len(candidates) >= 5:
                break

    return candidates


def inspect_sample_emails(
    email_members: list[tarfile.TarInfo],
) -> None:

    print()
    print("=" * 70)
    print("SAMPLE EMAILS")
    print("=" * 70)

    if not email_members:
        print("No likely email files detected.")
        return

    with tarfile.open(ARCHIVE_PATH, mode="r:gz") as archive:

        for index, member in enumerate(email_members, start=1):

            print()
            print(f"--- SAMPLE EMAIL {index} ---")
            print(f"Archive path: {member.name}")
            print(f"File size: {member.size:,} bytes")
            print()

            extracted = archive.extractfile(member)

            if extracted is None:
                print("Could not read archive member.")
                continue

            raw = extracted.read(12000)

            text = raw.decode(
                "utf-8",
                errors="replace",
            )

            lines = text.splitlines()

            for line in lines[:80]:
                print(line)


def main() -> None:
    files = inspect_archive()

    inspect_archive_structure(files)

    email_members = identify_email_members(files)

    print()
    print(
        f"Likely email files detected in sample scan: "
        f"{len(email_members)}"
    )

    inspect_sample_emails(email_members)

    print()
    print("=" * 70)
    print("INSPECTION COMPLETED WITHOUT FULL EXTRACTION")
    print("=" * 70)


if __name__ == "__main__":
    main()