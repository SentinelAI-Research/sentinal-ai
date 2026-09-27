from pathlib import Path

from ml.data.ami_normalizer import AMINormalizer

AMI_TRAIN_FILE = Path("data/raw/ami/train.jsonl")


def test_ami_normalizer_reads_real_dataset() -> None:
    records = AMINormalizer.normalize_file(AMI_TRAIN_FILE)

    assert records

    first_record = records[0]

    assert first_record.record_id
    assert first_record.source_dataset == "AMI Meeting Corpus"
    assert first_record.source_text
    assert first_record.reference_text
