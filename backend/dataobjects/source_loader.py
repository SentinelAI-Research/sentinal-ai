from pathlib import Path

from backend.dataobjects.models import DataObject
from backend.dataobjects.registry import DataObjectRegistry


class FileSourceLoader:
    """
    Loads a real local file and registers it as a provenance source.
    """

    def __init__(
        self,
        registry: DataObjectRegistry,
    ):
        self.registry = registry

    def load_text_file(
        self,
        file_path: str,
        object_id: str,
        sensitivity: str,
        created_step: str,
    ) -> DataObject:
        """
        Read a UTF-8 text file and register it as a source DataObject.
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Source path is not a file: {file_path}")

        content = path.read_text(encoding="utf-8")

        if not content.strip():
            raise ValueError(f"Source file is empty: {file_path}")

        return self.registry.register_source(
            object_id=object_id,
            source=str(path),
            content=content,
            sensitivity=sensitivity,
            created_step=created_step,
        )
