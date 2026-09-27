from __future__ import annotations

from pathlib import Path


DEFAULT_MEMORY_FILE = Path("data/sample/memory.txt")

# Backward-compatible alias used by existing code/tests.
MEMORY_FILE = DEFAULT_MEMORY_FILE


def write_memory(
    content: str,
    memory_path: str | Path | None = None,
) -> str:
    """
    Persist content to a local memory file.

    If memory_path is provided, that file is used for the memory
    operation. Otherwise, the existing default memory file is used.

    The optional memory_path allows individual trajectories to have
    isolated memory contexts without changing the existing API.
    """

    if not isinstance(content, str):
        raise TypeError(
            f"content must be str, got {type(content).__name__}"
        )

    if not content.strip():
        raise ValueError("Memory content cannot be empty.")

    path = Path(memory_path) if memory_path is not None else MEMORY_FILE

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(content.strip() + "\n")

    return content.strip()


def read_memory(
    memory_path: str | Path | None = None,
) -> str:
    """
    Read the complete contents of a local memory file.

    If memory_path is provided, that file is used. Otherwise, the
    existing default memory file is used.
    """

    path = Path(memory_path) if memory_path is not None else MEMORY_FILE

    if not path.exists():
        return ""

    return path.read_text(encoding="utf-8")