from backend.dataobjects.factory import create_data_object
from backend.provenance.graph import ProvenanceGraph


def test_mixed_source_lineage_reaches_original_sources():
    graph = ProvenanceGraph()

    source_a = create_data_object(
        object_id="source-a",
        source="dataset/a.txt",
        content="Information A",
        sensitivity="INTERNAL",
        transformation=None,
        parents=[],
        created_step="source-a",
    )

    source_b = create_data_object(
        object_id="source-b",
        source="dataset/b.txt",
        content="Information B",
        sensitivity="CONFIDENTIAL",
        transformation=None,
        parents=[],
        created_step="source-b",
    )

    combined = create_data_object(
        object_id="combined",
        source="tool:summarize",
        content="Combined information",
        sensitivity="CONFIDENTIAL",
        transformation="summarize",
        parents=["source-a", "source-b"],
        created_step="step-1",
    )

    derived = create_data_object(
        object_id="derived",
        source="tool:paraphrase",
        content="Paraphrased combined information",
        sensitivity="CONFIDENTIAL",
        transformation="paraphrase",
        parents=["combined"],
        created_step="step-2",
    )

    for obj in [source_a, source_b, combined, derived]:
        graph.add_object(obj)

    graph.add_relationship("source-a", "combined")
    graph.add_relationship("source-b", "combined")
    graph.add_relationship("combined", "derived")

    ancestors = graph.get_ancestors("derived")

    assert "combined" in ancestors
    assert "source-a" in ancestors
    assert "source-b" in ancestors

    assert graph.get_depth("combined") == 1
    assert graph.get_depth("derived") == 2
