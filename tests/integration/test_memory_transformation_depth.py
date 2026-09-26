from pathlib import Path

from benchmark.runners.memory_pipeline import (
    MemoryPipelineRunner,
)


def test_memory_chain_has_expected_transformation_depth():

    runner = MemoryPipelineRunner()

    result = runner.run(
        file_path=str(Path("data/sample/confidential_document.txt")),
        sensitivity="CONFIDENTIAL",
    )

    graph = result["runtime"].provenance_graph

    retrieval_id = result["retrieval"].object_id

    assert graph.get_transformation_depth(retrieval_id) == 4
