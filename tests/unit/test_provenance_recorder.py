from backend.dataobjects.models import DataObject
from backend.provenance.graph import ProvenanceGraph
from benchmark.runners.provenance_recorder import ProvenanceRecorder


def test_provenance_recorder_registers_parent_relationship():
    graph = ProvenanceGraph()
    recorder = ProvenanceRecorder(graph)

    source = DataObject(
        object_id="source-1",
        source="confidential.txt",
        sensitivity="CONFIDENTIAL",
        parents=[],
        transformation=None,
        content_hash="hash-source",
        embedding_ref=None,
        created_step="step-1",
    )

    summary = DataObject(
        object_id="summary-1",
        source="confidential.txt",
        sensitivity="CONFIDENTIAL",
        parents=["source-1"],
        transformation="summarize",
        content_hash="hash-summary",
        embedding_ref=None,
        created_step="step-2",
    )

    recorder.register_object(source)
    recorder.register_object(summary)

    assert graph.get_parents("summary-1") == ["source-1"]
    assert graph.get_depth("summary-1") == 1