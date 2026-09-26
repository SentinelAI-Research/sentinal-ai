from pathlib import Path

from backend.agent.proposal_builder import ProposalBuilder
from backend.dataobjects.memory_service import (
    MemoryDataObjectService,
)
from backend.dataobjects.registry import DataObjectRegistry
from backend.dataobjects.source_loader import FileSourceLoader
from backend.monitor.factory import create_sentinel_runtime


class MemoryPipelineRunner:
    """
    Executes a provenance-aware memory-mediated workflow.
    """

    def __init__(self):
        self.runtime = create_sentinel_runtime()

        self.registry = DataObjectRegistry(
            provenance_graph=self.runtime.provenance_graph
        )

        self.loader = FileSourceLoader(registry=self.registry)

        self.memory_service = MemoryDataObjectService(
            provenance_graph=self.runtime.provenance_graph
        )

        self.builder = ProposalBuilder(agent_id="memory-pipeline-agent")

    def run(
        self,
        file_path: str,
        sensitivity: str = "CONFIDENTIAL",
    ):
        path = Path(file_path)

        source = self.loader.load_text_file(
            file_path=str(path),
            object_id="memory-pipeline-source",
            sensitivity=sensitivity,
            created_step="source",
        )

        # ---------------------------------------------------------
        # 1. Read file
        # ---------------------------------------------------------

        read_proposal = self.builder.build(
            step_id="memory-read-file",
            tool_name="read_file",
            arguments={"path": str(path)},
            input_data_objects=[source.object_id],
        )

        read_decision, read_result = self.runtime.monitor_service.process(
            proposal=read_proposal,
            destination="INTERNAL",
        )

        if not read_result.executed:
            raise RuntimeError(f"read_file failed: {read_result.error}")

        # ---------------------------------------------------------
        # 2. Summarize
        # ---------------------------------------------------------

        summary_proposal = self.builder.build(
            step_id="memory-summary",
            tool_name="summarize",
            arguments={"text": read_result.output},
            input_data_objects=[read_result.output_data_object_id],
        )

        summary_decision, summary_result = self.runtime.monitor_service.process(
            proposal=summary_proposal,
            destination="INTERNAL",
        )

        if not summary_result.executed:
            raise RuntimeError(f"summarize failed: {summary_result.error}")

        summary_object = self.runtime.provenance_graph.get_object(
            summary_result.output_data_object_id
        )

        # ---------------------------------------------------------
        # 3. Write summary into memory
        # ---------------------------------------------------------

        memory_object = self.memory_service.create_memory_object(
            object_id="memory-pipeline-memory",
            content=summary_result.output,
            parent_object_ids=[summary_object.object_id],
            sensitivity=summary_object.sensitivity,
            created_step="memory-write",
        )

        # ---------------------------------------------------------
        # 4. Retrieve memory
        # ---------------------------------------------------------

        retrieval_object = self.memory_service.create_retrieval_object(
            object_id="memory-pipeline-retrieval",
            content=summary_result.output,
            memory_object_id=memory_object.object_id,
            sensitivity=memory_object.sensitivity,
            created_step="memory-read",
        )

        return {
            "runtime": self.runtime,
            "source": source,
            "summary": summary_object,
            "memory": memory_object,
            "retrieval": retrieval_object,
            "read_decision": read_decision,
            "summary_decision": summary_decision,
        }
