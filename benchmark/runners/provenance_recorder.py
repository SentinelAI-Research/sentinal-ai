from backend.dataobjects.models import DataObject
from backend.provenance.graph import ProvenanceGraph

"""
Scenario runner can eventually do :
Tool execution
      ↓
DataObject created
      ↓
ProvenanceRecorder
      ↓
ProvenanceGraph

The graph is therefore based on actual object creation.
Scenario Runner actually executes real tool operations and creates the corresponding DataObjects.
"""

class ProvenanceRecorder:
    def __init__(self, graph: ProvenanceGraph):
        self.graph = graph

    def register_object(
        self,
        data_object: DataObject,
    ) -> None:
        self.graph.add_object(data_object)

        for parent_id in data_object.parents:
            self.graph.add_relationship(
                parent_id,
                data_object.object_id,
            )