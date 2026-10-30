from backend.dataobjects.factory import create_data_object
from backend.dataobjects.models import DataObject
from backend.dataobjects.sensitivity import most_sensitive_sensitivity
from backend.provenance.graph import ProvenanceGraph


class MemoryDataObjectService:
    """
    Creates provenance-aware DataObjects for memory writes
    and memory retrievals.

    Memory is treated as a transformation layer rather than
    as a provenance reset.
    """

    def __init__(
        self,
        provenance_graph: ProvenanceGraph,
    ):
        self.provenance_graph = provenance_graph

    def create_memory_object(
        self,
        object_id: str,
        content: str,
        parent_object_ids: list[str],
        sensitivity: str,
        created_step: str,
    ) -> DataObject:
        """
        Create a DataObject representing data persisted into memory.
        """

        if not content:
            raise ValueError("Memory content cannot be empty.")

        if not parent_object_ids:
            raise ValueError("Memory object must have at least one parent.")

        parent_objects = self.provenance_graph.resolve_objects(parent_object_ids)
        object_sensitivity = most_sensitive_sensitivity(
            [parent.sensitivity for parent in parent_objects] + [sensitivity]
        )
        if self.provenance_graph.has_object(object_id):
            raise ValueError(f"Data object already exists: {object_id}")

        data_object = create_data_object(
            object_id=object_id,
            source="memory",
            content=content,
            sensitivity=object_sensitivity,
            transformation="write_memory",
            parents=parent_object_ids,
            created_step=created_step,
        )

        self.provenance_graph.add_object(data_object)

        for parent_id in parent_object_ids:
            self.provenance_graph.add_relationship(
                parent_id,
                object_id,
            )

        return data_object

    def create_retrieval_object(
        self,
        object_id: str,
        content: str,
        memory_object_id: str,
        sensitivity: str,
        created_step: str,
    ) -> DataObject:
        """
        Create a DataObject representing information retrieved
        from memory.
        """

        if not content:
            raise ValueError("Retrieved memory content cannot be empty.")

        memory_object = self.provenance_graph.get_object(memory_object_id)
        object_sensitivity = most_sensitive_sensitivity(
            [memory_object.sensitivity, sensitivity]
        )
        if self.provenance_graph.has_object(object_id):
            raise ValueError(f"Data object already exists: {object_id}")

        data_object = create_data_object(
            object_id=object_id,
            source="memory",
            content=content,
            sensitivity=object_sensitivity,
            transformation="read_memory",
            parents=[memory_object_id],
            created_step=created_step,
        )

        self.provenance_graph.add_object(data_object)

        self.provenance_graph.add_relationship(
            memory_object_id,
            object_id,
        )

        return data_object
