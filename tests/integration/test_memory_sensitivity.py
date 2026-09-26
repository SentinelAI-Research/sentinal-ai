from pathlib import Path

from benchmark.runners.memory_pipeline import (
    MemoryPipelineRunner,
)


def test_memory_does_not_downgrade_confidentiality():

    runner = MemoryPipelineRunner()

    result = runner.run(
        file_path=str(Path("data/sample/confidential_document.txt")),
        sensitivity="CONFIDENTIAL",
    )

    assert result["source"].sensitivity == "CONFIDENTIAL"

    assert result["summary"].sensitivity == "CONFIDENTIAL"

    assert result["memory"].sensitivity == "CONFIDENTIAL"

    assert result["retrieval"].sensitivity == "CONFIDENTIAL"
