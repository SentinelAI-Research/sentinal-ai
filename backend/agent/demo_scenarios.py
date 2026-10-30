from pathlib import Path
from uuid import uuid4

from backend.agent.deterministic_loop import (
    DeterministicAgentLoop,
    DeterministicTaskResult,
    PlannedAction,
)
from backend.dataobjects.registry import DataObjectRegistry
from backend.dataobjects.source_loader import FileSourceLoader
from backend.monitor.factory import create_sentinel_runtime

ROOT = Path(__file__).resolve().parents[2]
SAFE_SCENARIO_ID = "safe-summary"
BLOCKED_SCENARIO_ID = "confidential-external-message"

SCENARIO_CATALOG = [
    {
        "scenario_id": SAFE_SCENARIO_ID,
        "title": "Read, summarize, and draft",
        "description": "Read the public sample document, summarize it, then create a derived draft.",
    },
    {
        "scenario_id": BLOCKED_SCENARIO_ID,
        "title": "Confidential external transfer",
        "description": "Attempt a simulated external message using a confidential input; default policy should block it.",
    },
]


def run_demo_scenario(scenario_id: str) -> DeterministicTaskResult:
    if scenario_id not in {SAFE_SCENARIO_ID, BLOCKED_SCENARIO_ID}:
        raise ValueError("Unknown deterministic agent scenario.")

    run_id = str(uuid4())
    runtime = create_sentinel_runtime(
        policy_path=str(ROOT / "policies" / "default.yaml")
    )
    registry = DataObjectRegistry(runtime.provenance_graph)
    agent = DeterministicAgentLoop(runtime, agent_id=f"demo-agent-{run_id}")

    if scenario_id == SAFE_SCENARIO_ID:
        sample_path = ROOT / "data" / "sample" / "test.txt"
        source = FileSourceLoader(registry).load_text_file(
            file_path=str(sample_path),
            object_id=f"{run_id}-safe-source",
            sensitivity="PUBLIC",
            created_step=f"{run_id}-source",
        )
        actions = [
            PlannedAction(
                step_id=f"{run_id}-read",
                tool_name="read_file",
                destination="INTERNAL",
                description="Read the public sample document.",
                arguments={"path": str(sample_path)},
                input_data_objects=(source.object_id,),
            ),
            PlannedAction(
                step_id=f"{run_id}-summary",
                tool_name="summarize",
                destination="INTERNAL",
                description="Summarize the document returned by read_file.",
                arguments={"max_sentences": 3},
                input_from_previous_output=True,
                previous_output_argument="text",
            ),
            PlannedAction(
                step_id=f"{run_id}-draft",
                tool_name="paraphrase",
                destination="INTERNAL",
                description="Create a derived draft from the summary.",
                arguments={},
                input_from_previous_output=True,
                previous_output_argument="text",
            ),
        ]
        description = "Read the public sample, summarize it, and create an internal derived draft."
    else:
        registry.register_source(
            object_id=f"{run_id}-confidential-source",
            source="demo-fixture:confidential-record",
            content="Synthetic confidential fixture for the blocked-transfer demonstration.",
            sensitivity="CONFIDENTIAL",
            created_step=f"{run_id}-source",
        )
        actions = [
            PlannedAction(
                step_id=f"{run_id}-send",
                tool_name="send_message",
                destination="EXTERNAL",
                description="Request a simulated external message using confidential data.",
                arguments={
                    "recipient": "demo-recipient",
                    "message": "Controlled blocked-transfer demonstration.",
                },
                input_data_objects=(f"{run_id}-confidential-source",),
            )
        ]
        description = (
            "Demonstrate default-policy blocking of confidential data sent externally."
        )

    return agent.run(description, actions, run_id=run_id)
