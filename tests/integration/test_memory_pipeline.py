from pathlib import Path

from benchmark.runners.memory_pipeline import (
    MemoryPipelineRunner,
)


def test_memory_pipeline_preserves_lineage():

    runner = MemoryPipelineRunner()

    result = runner.run(
        file_path=str(Path("data/sample/confidential_document.txt")),
        sensitivity="CONFIDENTIAL",
    )

    runtime = result["runtime"]

    source = result["source"]
    summary = result["summary"]
    memory = result["memory"]
    retrieval = result["retrieval"]

    # ---------------------------------------------------------
    # Verify security decisions
    # ---------------------------------------------------------

    assert result["read_decision"].verdict == "ALLOW"

    assert result["summary_decision"].verdict == "ALLOW"

    # ---------------------------------------------------------
    # Verify sensitivity is preserved
    # ---------------------------------------------------------

    assert source.sensitivity == ("CONFIDENTIAL")

    assert summary.sensitivity == ("CONFIDENTIAL")

    assert memory.sensitivity == ("CONFIDENTIAL")

    assert retrieval.sensitivity == ("CONFIDENTIAL")

    # ---------------------------------------------------------
    # Verify immediate parent relationships
    # ---------------------------------------------------------

    assert summary.parents == ["memory-read-file-output"]

    assert memory.parents == [summary.object_id]

    assert retrieval.parents == [memory.object_id]

    # ---------------------------------------------------------
    # Verify complete provenance lineage
    # ---------------------------------------------------------

    graph = runtime.provenance_graph

    retrieval_ancestors = graph.get_ancestors(retrieval.object_id)

    assert source.object_id in retrieval_ancestors

    assert summary.object_id in retrieval_ancestors

    assert memory.object_id in retrieval_ancestors

    # ---------------------------------------------------------
    # Verify provenance depth
    # ---------------------------------------------------------

    assert graph.get_depth(retrieval.object_id) == 4
