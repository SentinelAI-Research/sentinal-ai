from backend.dataobjects.factory import create_data_object
from backend.dataobjects.models import DataObject
from backend.provenance.graph import ProvenanceGraph


class DataObjectRegistry:
    """
    Registers original source data as provenance roots.
    """

    def __init__(
        self,
        provenance_graph: ProvenanceGraph,
    ):
        self.provenance_graph = provenance_graph

    def register_source(
        self,
        object_id: str,
        source: str,
        content: str,
        sensitivity: str,
        created_step: str,
    ) -> DataObject:
        """
        Register a source DataObject with no parents.
        """

        data_object = create_data_object(
            object_id=object_id,
            source=source,
            content=content,
            sensitivity=sensitivity,
            transformation=None,
            parents=[],
            created_step=created_step,
        )

        self.provenance_graph.add_object(data_object)

        return data_object
