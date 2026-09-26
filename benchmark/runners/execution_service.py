from pathlib import Path

from backend.dataobjects.factory import create_data_object
from backend.tools.file_tools import read_file, summarize
from backend.tools.memory_tools import write_memory, read_memory
from backend.tools.communication_tools import send_message
from backend.trajectory.store import TrajectoryStore
from benchmark.runners.provenance_recorder import ProvenanceRecorder
from benchmark.runners.scenario_runner import ScenarioRunner
from benchmark.scenarios.models import SecurityScenario

"""
REAL FILE
   ↓
read_file()
   ↓
DataObject
   ↓
TrajectoryStep
   ↓
summarize()
   ↓
NEW DataObject
   ↓
Provenance edge
   ↓
NEW TrajectoryStep
"""
class ScenarioExecutionService:
    def __init__(
        self,
        trajectory_store: TrajectoryStore,
        provenance_recorder: ProvenanceRecorder,
    ):
        self.trajectory_store = trajectory_store
        self.provenance_recorder = provenance_recorder
        self.runner = ScenarioRunner(trajectory_store)

    def execute_summary(
        self,
        scenario: SecurityScenario,
        file_path: str,
        sensitivity: str,
    ):
        session_id = f"session-{scenario.scenario_id}"

        content = read_file(file_path)

        source_object = create_data_object(
            object_id=f"{scenario.scenario_id}-source",
            source=file_path,
            content=content,
            sensitivity=sensitivity,
            transformation=None,
            parents=[],
            created_step="step-1",
        )

        self.provenance_recorder.register_object(source_object)

        self.runner.record_step(
            scenario=scenario,
            step_id="step-1",
            agent_id="benchmark-agent",
            session_id=session_id,
            tool_name="read_file",
            arguments={"path": file_path},
            output_data_objects=[source_object.object_id],
        )

        summary = summarize(content)

        summary_object = create_data_object(
            object_id=f"{scenario.scenario_id}-summary",
            source=file_path,
            content=summary,
            sensitivity=sensitivity,
            transformation="summarize",
            parents=[source_object.object_id],
            created_step="step-2",
        )

        self.provenance_recorder.register_object(summary_object)

        self.runner.record_step(
            scenario=scenario,
            step_id="step-2",
            agent_id="benchmark-agent",
            session_id=session_id,
            tool_name="summarize",
            arguments={},
            input_data_objects=[source_object.object_id],
            output_data_objects=[summary_object.object_id],
        )

        return self.runner.get_execution(
            scenario,
            session_id,
        )

    def execute_memory_path(
        self,
        scenario: SecurityScenario,
        file_path: str,
        sensitivity: str,
    ):
        session_id = f"session-{scenario.scenario_id}"

        content = read_file(file_path)

        source_object = create_data_object(
            object_id=f"{scenario.scenario_id}-source",
            source=file_path,
            content=content,
            sensitivity=sensitivity,
            transformation=None,
            parents=[],
            created_step="step-1",
        )

        self.provenance_recorder.register_object(source_object)

        self.runner.record_step(
            scenario=scenario,
            step_id="step-1",
            agent_id="benchmark-agent",
            session_id=session_id,
            tool_name="read_file",
            arguments={"path": file_path},
            output_data_objects=[source_object.object_id],
        )

        summary = summarize(content)

        summary_object = create_data_object(
            object_id=f"{scenario.scenario_id}-summary",
            source=file_path,
            content=summary,
            sensitivity=sensitivity,
            transformation="summarize",
            parents=[source_object.object_id],
            created_step="step-2",
        )

        self.provenance_recorder.register_object(summary_object)

        self.runner.record_step(
            scenario=scenario,
            step_id="step-2",
            agent_id="benchmark-agent",
            session_id=session_id,
            tool_name="summarize",
            arguments={},
            input_data_objects=[source_object.object_id],
            output_data_objects=[summary_object.object_id],
        )

        write_memory(summary)

        memory_object = create_data_object(
            object_id=f"{scenario.scenario_id}-memory",
            source="memory",
            content=summary,
            sensitivity=sensitivity,
            transformation="write_memory",
            parents=[summary_object.object_id],
            created_step="step-3",
        )

        self.provenance_recorder.register_object(memory_object)

        self.runner.record_step(
            scenario=scenario,
            step_id="step-3",
            agent_id="benchmark-agent",
            session_id=session_id,
            tool_name="write_memory",
            arguments={},
            input_data_objects=[summary_object.object_id],
            output_data_objects=[memory_object.object_id],
        )

        memory_content = read_memory()

        retrieved_object = create_data_object(
            object_id=f"{scenario.scenario_id}-memory-retrieval",
            source="memory",
            content=memory_content,
            sensitivity=sensitivity,
            transformation="read_memory",
            parents=[memory_object.object_id],
            created_step="step-4",
        )

        self.provenance_recorder.register_object(retrieved_object)

        self.runner.record_step(
            scenario=scenario,
            step_id="step-4",
            agent_id="benchmark-agent",
            session_id=session_id,
            tool_name="read_memory",
            arguments={},
            input_data_objects=[memory_object.object_id],
            output_data_objects=[retrieved_object.object_id],
        )

        self.runner.record_step(
            scenario=scenario,
            step_id="step-5",
            agent_id="benchmark-agent",
            session_id=session_id,
            tool_name="send_message",
            arguments={
                "recipient": "external@example.com",
            },
            input_data_objects=[retrieved_object.object_id],
        )

        return self.runner.get_execution(
            scenario,
            session_id,
        )