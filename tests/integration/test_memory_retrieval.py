from backend.dataobjects.factory import create_data_object
from backend.dataobjects.memory_service import (
    MemoryDataObjectService,
)
from backend.provenance.graph import ProvenanceGraph


def test_memory_retrieval_preserves_full_lineage():

    graph = ProvenanceGraph()

    source = create_data_object(
        object_id="source",
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
        object_id="memory-object",
        content="Confidential information.",
        parent_object_ids=["source"],
        sensitivity="CONFIDENTIAL",
        created_step="write-memory",
    )

    retrieval_object = memory_service.create_retrieval_object(
        object_id="retrieval-object",
        content="Confidential information.",
        memory_object_id=memory_object.object_id,
        sensitivity="CONFIDENTIAL",
        created_step="read-memory",
    )

    assert retrieval_object.parents == ["memory-object"]

    assert retrieval_object.sensitivity == ("CONFIDENTIAL")

    assert retrieval_object.transformation == ("read_memory")

    assert graph.get_depth("retrieval-object") == 2

    ancestors = graph.get_ancestors("retrieval-object")

    assert "source" in ancestors
    assert "memory-object" in ancestors
