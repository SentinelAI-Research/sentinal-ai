from pathlib import Path

from benchmark.runners.memory_pipeline import (
    MemoryPipelineRunner,
)


def test_memory_chain_contains_original_source():

    runner = MemoryPipelineRunner()

    result = runner.run(
        file_path=str(Path("data/sample/confidential_document.txt")),
        sensitivity="CONFIDENTIAL",
    )

    graph = result["runtime"].provenance_graph

    retrieval_id = result["retrieval"].object_id

    ancestors = graph.get_ancestors(retrieval_id)

    assert result["source"].object_id in ancestors

    assert result["summary"].object_id in ancestors

    assert result["memory"].object_id in ancestors

    assert graph.get_depth(retrieval_id) == 4
