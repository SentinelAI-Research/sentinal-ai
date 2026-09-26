from backend.dataobjects.factory import create_data_object
from backend.dataobjects.memory_service import (
    MemoryDataObjectService,
)
from backend.provenance.graph import ProvenanceGraph


def test_memory_object_preserves_parent_provenance():

    graph = ProvenanceGraph()

    source = create_data_object(
        object_id="memory-source",
        source="test-source",
        content="Confidential information.",
        sensitivity="CONFIDENTIAL",
        transformation=None,
        parents=[],
        created_step="source",
    )

    graph.add_object(source)

    memory_service = MemoryDataObjectService(provenance_graph=graph)

    memory_object = memory_service.create_memory_object(
        object_id="memory-write",
        content="Confidential information.",
        parent_object_ids=["memory-source"],
        sensitivity="CONFIDENTIAL",
        created_step="write-memory",
    )

    assert memory_object.parents == ["memory-source"]

    assert memory_object.sensitivity == ("CONFIDENTIAL")

    assert memory_object.transformation == ("write_memory")

    assert graph.get_parents("memory-write") == ["memory-source"]

    assert graph.get_depth("memory-write") == 1
