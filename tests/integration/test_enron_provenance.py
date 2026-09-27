from pathlib import Path

from backend.provenance.graph import ProvenanceGraph
from ml.data.enron_provenance import EnronProvenanceLoader


ARCHIVE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "enron"
    / "enron_mail_20150507.tar.gz"
)


def test_enron_archive_creates_root_provenance_objects():
    graph, records = EnronProvenanceLoader.load_archive(
        archive_path=ARCHIVE_PATH,
        max_records=5,
    )

    assert isinstance(graph, ProvenanceGraph)

    assert len(records) == 5

    assert len(graph.graph.nodes) == 5

    for record in records:
        object_id = f"enron-{record.record_id}"

        assert object_id in graph.graph

        data_object = graph.get_object(object_id)

        assert data_object.object_id == object_id
        assert data_object.source.startswith("Enron Email Dataset:")
        assert data_object.sensitivity == "PUBLIC"
        assert data_object.transformation == "source_ingestion"
        assert data_object.parents == []

        assert graph.get_parents(object_id) == []
        assert graph.get_children(object_id) == []

        assert graph.get_depth(object_id) == 0

        assert graph.get_transformation_depth(object_id) == 1

        assert len(data_object.content_hash) == 64