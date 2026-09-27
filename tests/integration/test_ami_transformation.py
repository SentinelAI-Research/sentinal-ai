import json
from pathlib import Path

from ml.data.ami_normalizer import AMINormalizer
from ml.data.ami_provenance import AMIProvenanceLoader
from ml.data.ami_transformation import AMITransformationService

AMI_TRAIN_FILE = Path("data/raw/ami/train.jsonl")


def test_real_ami_summarization_creates_provenance_child() -> None:
    loader = AMIProvenanceLoader()

    graph, objects = loader.load_split(
        input_path=AMI_TRAIN_FILE,
        split_name="train",
        max_records=1,
    )

    source_object = objects[0]

    # Read the same real AMI record used to create the DataObject.
    with AMI_TRAIN_FILE.open("r", encoding="utf-8") as file:
        first_line = file.readline().strip()

    raw_record = json.loads(first_line)

    record = AMINormalizer.normalize_record(raw_record)

    assert record.record_id == "30"
    assert record.source_text.strip()

    summary_object = AMITransformationService.summarize_object(
        graph=graph,
        source_object=source_object,
        source_text=record.source_text,
        step_id="ami-summary-step-1",
    )

    assert summary_object.object_id == "ami-train-30-summary"
    assert summary_object.transformation == "summarize"
    assert summary_object.parents == ["ami-train-30"]
    assert summary_object.created_step == "ami-summary-step-1"

    assert summary_object.content_hash
    assert summary_object.content_hash != source_object.content_hash

    assert graph.get_parents(summary_object.object_id) == [source_object.object_id]

    assert graph.get_children(source_object.object_id) == [summary_object.object_id]

    assert graph.get_depth(summary_object.object_id) == 1

    # The source object has "source_ingestion" as its transformation,
    # and the derived object has "summarize".
    # Therefore the transformation depth is 2.
    assert graph.get_transformation_depth(summary_object.object_id) == 2
