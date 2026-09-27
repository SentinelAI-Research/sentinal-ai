from pathlib import Path

from backend.provenance.graph import ProvenanceGraph
from ml.data.ami_provenance import AMIProvenanceLoader

AMI_TRAIN_FILE = Path("data/raw/ami/train.jsonl")


def test_real_ami_records_enter_provenance_graph() -> None:
    loader = AMIProvenanceLoader()

    graph, objects = loader.load_split(
        input_path=AMI_TRAIN_FILE,
        split_name="train",
        max_records=5,
    )

    assert isinstance(
        graph,
        ProvenanceGraph,
    )

    assert len(objects) == 5

    for data_object in objects:
        assert graph.get_object(data_object.object_id) == data_object

        assert graph.get_parents(data_object.object_id) == []

        assert graph.get_children(data_object.object_id) == []

        assert graph.get_depth(data_object.object_id) == 0
