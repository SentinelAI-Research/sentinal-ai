from pathlib import Path

from backend.dataobjects.models import DataObject
from backend.provenance.graph import ProvenanceGraph
from ml.data.ami_ingestion import AMIIngestionPipeline


class AMIProvenanceLoader:
    """
    Loads real AMI records into a SentinelAI provenance graph.

    Each original AMI record becomes a root DataObject because
    it has no parent object inside the SentinelAI pipeline.
    """

    def __init__(self) -> None:
        self.pipeline = AMIIngestionPipeline()

    def load_split(
        self,
        input_path: str | Path,
        split_name: str,
        max_records: int | None = None,
    ) -> tuple[
        ProvenanceGraph,
        list[DataObject],
    ]:
        data_objects = self.pipeline.ingest_file(
            input_path=input_path,
            split_name=split_name,
            max_records=max_records,
        )

        graph = ProvenanceGraph()

        for data_object in data_objects:
            graph.add_object(data_object)

        return graph, data_objects
