from backend.dataobjects.models import DataObject
from backend.provenance.graph import ProvenanceGraph
from ml.features.provenance_features import extract_provenance_features


def test_provenance_features_are_derived_from_graph():
    graph = ProvenanceGraph()

    source = DataObject(
        object_id="source",
        source="test.txt",
        sensitivity="CONFIDENTIAL",
        parents=[],
        transformation=None,
        content_hash="hash-source",
        embedding_ref=None,
        created_step="step-1",
    )

    summary = DataObject(
        object_id="summary",
        source="test.txt",
        sensitivity="CONFIDENTIAL",
        parents=["source"],
        transformation="summarize",
        content_hash="hash-summary",
        embedding_ref=None,
        created_step="step-2",
    )

    paraphrase = DataObject(
        object_id="paraphrase",
        source="test.txt",
        sensitivity="CONFIDENTIAL",
        parents=["summary"],
        transformation="paraphrase",
        content_hash="hash-paraphrase",
        embedding_ref=None,
        created_step="step-3",
    )

    graph.add_object(source)
    graph.add_object(summary)
    graph.add_object(paraphrase)

    graph.add_relationship("source", "summary")
    graph.add_relationship("summary", "paraphrase")

    features = extract_provenance_features(
        graph,
        "paraphrase",
    )

    assert features["provenance_depth"] == 2
    assert features["transformation_count"] == 2

def test_provenance_graph_returns_data_object():
    graph = ProvenanceGraph()

    source = DataObject(
        object_id="source",
        source="test.txt",
        sensitivity="CONFIDENTIAL",
        parents=[],
        transformation=None,
        content_hash="hash-source",
        embedding_ref=None,
        created_step="step-1",
    )

    graph.add_object(source)

    result = graph.get_object("source")

    assert result.object_id == "source"
    assert result.sensitivity == "CONFIDENTIAL"