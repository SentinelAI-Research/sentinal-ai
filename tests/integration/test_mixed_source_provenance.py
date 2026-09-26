from backend.dataobjects.factory import create_data_object
from backend.provenance.graph import ProvenanceGraph


def test_mixed_source_provenance_preserves_all_parents():
    graph = ProvenanceGraph()

    source_a = create_data_object(
        object_id="source-a",
        source="dataset/document_a.txt",
        content="Internal information from source A.",
        sensitivity="INTERNAL",
        transformation=None,
        parents=[],
        created_step="source-a",
    )

    source_b = create_data_object(
        object_id="source-b",
        source="dataset/document_b.txt",
        content="Confidential information from source B.",
        sensitivity="CONFIDENTIAL",
        transformation=None,
        parents=[],
        created_step="source-b",
    )

    combined = create_data_object(
        object_id="combined-object",
        source="tool:summarize",
        content="Combined information from both sources.",
        sensitivity="CONFIDENTIAL",
        transformation="summarize",
        parents=["source-a", "source-b"],
        created_step="step-3",
    )

    graph.add_object(source_a)
    graph.add_object(source_b)
    graph.add_object(combined)

    graph.add_relationship("source-a", "combined-object")
    graph.add_relationship("source-b", "combined-object")

    assert combined.parents == ["source-a", "source-b"]

    ancestors = graph.get_ancestors("combined-object")

    assert "source-a" in ancestors
    assert "source-b" in ancestors

    assert graph.get_depth("source-a") == 0
    assert graph.get_depth("source-b") == 0
    assert graph.get_depth("combined-object") == 1
