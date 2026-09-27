from backend.dataobjects.factory import create_data_object
from backend.dataobjects.models import DataObject
from backend.provenance.graph import ProvenanceGraph
from backend.tools.file_tools import summarize


class AMITransformationService:
    """
    Applies a real transformation to text originating from an AMI
    DataObject and records the resulting DataObject in the provenance graph.

    The DataObject deliberately stores metadata and a content hash rather
    than the raw text. Therefore, the source text is supplied separately
    to the transformation.
    """

    @staticmethod
    def summarize_object(
        graph: ProvenanceGraph,
        source_object: DataObject,
        source_text: str,
        step_id: str,
    ) -> DataObject:
        """
        Summarize AMI source text and create a provenance-linked child object.
        """

        if source_object.object_id not in graph.graph:
            raise ValueError(
                f"Source object '{source_object.object_id}' "
                "does not exist in the provenance graph."
            )

        if not source_text.strip():
            raise ValueError("Source text cannot be empty.")

        summary_result = summarize(text=source_text)

        if not isinstance(summary_result, str):
            raise TypeError("Summarization result must be a string.")

        if not summary_result.strip():
            raise ValueError("Summarization produced empty output.")

        output_object_id = f"{source_object.object_id}-summary"

        output_object = create_data_object(
            object_id=output_object_id,
            source=source_object.source,
            content=summary_result,
            sensitivity=source_object.sensitivity,
            transformation="summarize",
            parents=[source_object.object_id],
            created_step=step_id,
        )

        graph.add_object(output_object)

        graph.add_relationship(
            source_object.object_id,
            output_object.object_id,
        )

        return output_object
