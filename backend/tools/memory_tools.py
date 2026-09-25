from pathlib import Path


MEMORY_FILE = Path("data/sample/memory.txt")


def write_memory(content: str) -> str:
    """
    Persist information into SentinelAI's local memory store.
    """

    if not content.strip():
        raise ValueError("Memory content cannot be empty.")

    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with MEMORY_FILE.open("a", encoding="utf-8") as file:
        file.write(content.strip() + "\n")

    return content.strip()


def read_memory() -> str:
    """
    Read all information currently stored in local memory.
    """

    if not MEMORY_FILE.exists():
        return ""

    return MEMORY_FILE.read_text(encoding="utf-8")