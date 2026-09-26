from pathlib import Path

from backend.dataobjects.registry import DataObjectRegistry
from backend.dataobjects.source_loader import FileSourceLoader
from backend.provenance.graph import ProvenanceGraph


def test_real_confidential_file_becomes_source_object():

    graph = ProvenanceGraph()

    registry = DataObjectRegistry(provenance_graph=graph)

    loader = FileSourceLoader(registry=registry)

    file_path = Path("data/sample/confidential_document.txt")

    data_object = loader.load_text_file(
        file_path=str(file_path),
        object_id="confidential-document-source",
        sensitivity="CONFIDENTIAL",
        created_step="source",
    )

    assert data_object.object_id == ("confidential-document-source")

    assert data_object.sensitivity == "CONFIDENTIAL"

    assert data_object.parents == []

    assert data_object.transformation is None

    assert data_object.content_hash

    assert (
        graph.get_object("confidential-document-source").content_hash
        == data_object.content_hash
    )
