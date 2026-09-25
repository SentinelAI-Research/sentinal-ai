import networkx as nx

from backend.dataobjects.models import DataObject


class ProvenanceGraph:
    """
    Maintains the lineage of DataObjects flowing through SentinelAI.

    Each DataObject is represented as a node.
    A directed edge from parent -> child represents a
    transformation or derivation.

    For : file_001
            │
            ▼
          summary_001
            │
            ▼
          paraphrase_001, 

    Graph will store : file_001 → summary_001
                       summary_001 → paraphrase_001
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_object(self, data_object: DataObject) -> None:
        """
        Add a DataObject to the provenance graph.
        """

        self.graph.add_node(
            data_object.object_id,
            data_object=data_object,
        )

    def add_relationship(
        self,
        parent_id: str,
        child_id: str,
    ) -> None:
        """
        Record a parent -> child provenance relationship.
        """

        if parent_id not in self.graph:
            raise ValueError(f"Parent object not found: {parent_id}")

        if child_id not in self.graph:
            raise ValueError(f"Child object not found: {child_id}")

        self.graph.add_edge(
            parent_id,
            child_id,
        )

    def get_parents(self, object_id: str) -> list[str]:
        """
        Return the direct parents of a DataObject.
        """

        if object_id not in self.graph:
            raise ValueError(f"Object not found: {object_id}")

        return list(self.graph.predecessors(object_id))

    def get_children(self, object_id: str) -> list[str]:
        """
        Return the direct children of a DataObject.
        """

        if object_id not in self.graph:
            raise ValueError(f"Object not found: {object_id}")

        return list(self.graph.successors(object_id))

    def get_ancestors(self, object_id: str) -> list[str]:
        """
        Return all ancestors of a DataObject.

        Ancestors include direct parents as well as objects
        further upstream in the provenance graph.
        """

        if object_id not in self.graph:
            raise ValueError(f"Object not found: {object_id}")

        return list(nx.ancestors(self.graph, object_id))

    def get_path(self, source_id: str, target_id: str) -> list[str]:
        """
        Return the provenance path from a source object
        to a derived target object.
        """

        if source_id not in self.graph:
            raise ValueError(f"Source object not found: {source_id}")

        if target_id not in self.graph:
            raise ValueError(f"Target object not found: {target_id}")

        try:
            return nx.shortest_path(
                self.graph,
                source_id,
                target_id,
            )
        except nx.NetworkXNoPath as exc:
            raise ValueError(
                f"No provenance path exists from {source_id} "
                f"to {target_id}"
            ) from exc