from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from backend.dataobjects.factory import create_data_object
from backend.provenance.graph import ProvenanceGraph
from backend.tools.communication_tools import send_message
from backend.tools.file_tools import paraphrase
from backend.tools.memory_tools import read_memory, write_memory
from ml.data.ami_normalizer import AMINormalizer
from ml.data.ami_transformation import AMITransformationService


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "raw"
    / "ami"
    / "train.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "ami"
    / "ami_security_experiment.jsonl"
)

TOTAL_TRAJECTORIES = 100
STEPS_PER_TRAJECTORY = 6


SCENARIO_ID = "ami-memory-mediated-leakage"
SCENARIO_TYPE = "LONG_HORIZON_LEAKAGE"


STEP_LABELS = {
    1: "SAFE",
    2: "EARLY_RISK",
    3: "EARLY_RISK",
    4: "EARLY_RISK",
    5: "EARLY_RISK",
    6: "VIOLATION",
}


TOOL_NAMES = {
    1: "read_file",
    2: "summarize",
    3: "write_memory",
    4: "read_memory",
    5: "paraphrase",
    6: "send_message",
}


TRANSFORMATIONS = {
    1: "source_ingestion",
    2: "summarize",
    3: "write_memory",
    4: "read_memory",
    5: "paraphrase",
    6: "send_message",
}


def load_unique_ami_records(
    input_path: Path,
    max_records: int,
) -> list:
    """
    Load unique AMI records from the real training split.

    Deduplication is performed using the AMI dataset's own record ID.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            f"AMI input dataset does not exist: {input_path}"
        )

    if max_records <= 0:
        raise ValueError(
            "max_records must be greater than zero."
        )

    records = []
    seen_ids: set[str] = set()

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            if len(records) >= max_records:
                break

            line = line.strip()

            if not line:
                continue

            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} "
                    f"in {input_path}: {exc}"
                ) from exc

            normalized = AMINormalizer.normalize_record(
                raw_record
            )

            if normalized.record_id in seen_ids:
                continue

            seen_ids.add(normalized.record_id)
            records.append(normalized)

    if not records:
        raise RuntimeError(
            "No valid unique AMI records were found."
        )

    return records


def create_source_object(
    record,
    graph: ProvenanceGraph,
    trajectory_number: int,
):
    """
    Create the root DataObject representing the real AMI dialogue.
    """

    object_id = (
        f"ami-trajectory-{trajectory_number}-"
        f"{record.record_id}"
    )

    source_object = create_data_object(
        object_id=object_id,
        source=(
            "AMI Meeting Corpus:"
            f"{record.record_id}"
        ),
        content=record.source_text,
        sensitivity="PUBLIC",
        transformation="source_ingestion",
        parents=[],
        created_step=(
            f"ami-trajectory-{trajectory_number}-step-1"
        ),
    )

    graph.add_object(source_object)

    return source_object


def build_trajectory(
    record,
    trajectory_number: int,
) -> tuple[ProvenanceGraph, list[str]]:
    """
    Build one six-step memory-mediated trajectory from
    a real AMI dialogue.

    Steps:

    1. read_file
    2. summarize
    3. write_memory
    4. read_memory
    5. paraphrase
    6. send_message
    """

    graph = ProvenanceGraph()

    source_object = create_source_object(
        record=record,
        graph=graph,
        trajectory_number=trajectory_number,
    )

    source_id = source_object.object_id

    # ---------------------------------------------------------
    # Step 2 — Summarize
    # ---------------------------------------------------------

    summary_object = AMITransformationService.summarize_object(
        graph=graph,
        source_object=source_object,
        source_text=record.source_text,
        step_id=(
            f"ami-trajectory-{trajectory_number}-step-2"
        ),
    )

    # The AMI transformation service deliberately keeps raw
    # text outside the DataObject, so we retain the actual
    # transformation result separately for the next operation.
    from backend.tools.file_tools import summarize

    summary_text = summarize(
        text=record.source_text
    )

    # ---------------------------------------------------------
    # Steps 3–4 use isolated temporary memory
    # ---------------------------------------------------------

    with tempfile.TemporaryDirectory(
        prefix="sentinelai-ami-memory-"
    ) as temp_dir:

        memory_path = (
            Path(temp_dir) / "memory.txt"
        )

        # Step 3 — Write memory
        memory_text = write_memory(
            content=summary_text,
            memory_path=memory_path,
        )

        memory_object = create_data_object(
            object_id=(
                f"{summary_object.object_id}-memory"
            ),
            source=source_object.source,
            content=memory_text,
            sensitivity=summary_object.sensitivity,
            transformation="write_memory",
            parents=[
                summary_object.object_id
            ],
            created_step=(
                f"ami-trajectory-{trajectory_number}-step-3"
            ),
        )

        graph.add_object(memory_object)

        graph.add_relationship(
            summary_object.object_id,
            memory_object.object_id,
        )

        # Step 4 — Read memory
        retrieved_text = read_memory(
            memory_path=memory_path
        )

        retrieved_object = create_data_object(
            object_id=(
                f"{memory_object.object_id}-retrieved"
            ),
            source=source_object.source,
            content=retrieved_text,
            sensitivity=memory_object.sensitivity,
            transformation="read_memory",
            parents=[
                memory_object.object_id
            ],
            created_step=(
                f"ami-trajectory-{trajectory_number}-step-4"
            ),
        )

        graph.add_object(retrieved_object)

        graph.add_relationship(
            memory_object.object_id,
            retrieved_object.object_id,
        )

    # ---------------------------------------------------------
    # Step 5 — Paraphrase
    # ---------------------------------------------------------

    paraphrased_text = paraphrase(
        text=retrieved_text
    )

    paraphrase_object = create_data_object(
        object_id=(
            f"{retrieved_object.object_id}-paraphrase"
        ),
        source=source_object.source,
        content=paraphrased_text,
        sensitivity=retrieved_object.sensitivity,
        transformation="paraphrase",
        parents=[
            retrieved_object.object_id
        ],
        created_step=(
            f"ami-trajectory-{trajectory_number}-step-5"
        ),
    )

    graph.add_object(paraphrase_object)

    graph.add_relationship(
        retrieved_object.object_id,
        paraphrase_object.object_id,
    )

    # ---------------------------------------------------------
    # Step 6 — Send message
    # ---------------------------------------------------------

    message_result = send_message(
        recipient="controlled-local-sink",
        message=paraphrased_text,
    )

    if not isinstance(message_result, str):
        message_result = str(message_result)

    message_object = create_data_object(
        object_id=(
            f"{paraphrase_object.object_id}-message"
        ),
        source=source_object.source,
        content=message_result,
        sensitivity=paraphrase_object.sensitivity,
        transformation="send_message",
        parents=[
            paraphrase_object.object_id
        ],
        created_step=(
            f"ami-trajectory-{trajectory_number}-step-6"
        ),
    )

    graph.add_object(message_object)

    graph.add_relationship(
        paraphrase_object.object_id,
        message_object.object_id,
    )

    trajectory = [
        source_id,
        summary_object.object_id,
        memory_object.object_id,
        retrieved_object.object_id,
        paraphrase_object.object_id,
        message_object.object_id,
    ]

    if len(trajectory) != STEPS_PER_TRAJECTORY:
        raise RuntimeError(
            "Generated trajectory does not contain "
            f"{STEPS_PER_TRAJECTORY} steps."
        )

    return graph, trajectory


def build_output_record(
    graph: ProvenanceGraph,
    trajectory: list[str],
    trajectory_id: str,
    step_number: int,
) -> dict:

    object_id = trajectory[step_number - 1]

    data_object = graph.get_object(
        object_id
    )

    if data_object is None:
        raise RuntimeError(
            f"DataObject '{object_id}' was not found."
        )

    parents = graph.get_parents(
        object_id
    )

    return {
        "trajectory_id": trajectory_id,
        "scenario_id": SCENARIO_ID,
        "scenario_type": SCENARIO_TYPE,
        "step_number": step_number,
        "total_steps": STEPS_PER_TRAJECTORY,
        "tool_name": TOOL_NAMES[step_number],
        "input_object_ids": parents,
        "output_object_id": object_id,
        "label": STEP_LABELS[step_number],
        "provenance_depth": graph.get_depth(
            object_id
        ),
        "transformation_depth": (
            graph.get_transformation_depth(
                object_id
            )
        ),
        "source": data_object.source,
        "sensitivity": data_object.sensitivity,
        "transformation": TRANSFORMATIONS[
            step_number
        ],
        "content_hash": data_object.content_hash,
    }


def main() -> None:
    print("=" * 70)
    print("SENTINELAI — AMI SECURITY EXPERIMENT DATASET")
    print("=" * 70)

    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()

    records = load_unique_ami_records(
        input_path=INPUT_PATH,
        max_records=TOTAL_TRAJECTORIES,
    )

    if len(records) < TOTAL_TRAJECTORIES:
        raise RuntimeError(
            f"Requested {TOTAL_TRAJECTORIES} unique AMI records, "
            f"but only {len(records)} were available."
        )

    print(
        f"Loaded {len(records)} unique AMI records."
    )
    print()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_records = 0

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for trajectory_number, record in enumerate(
            records[:TOTAL_TRAJECTORIES],
            start=1,
        ):

            try:
                graph, trajectory = build_trajectory(
                    record=record,
                    trajectory_number=trajectory_number,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed while building AMI trajectory "
                    f"{trajectory_number} "
                    f"for record '{record.record_id}'."
                ) from exc

            trajectory_id = (
                f"ami-trajectory-{trajectory_number}"
            )

            for step_number in range(
                1,
                STEPS_PER_TRAJECTORY + 1,
            ):

                output_record = build_output_record(
                    graph=graph,
                    trajectory=trajectory,
                    trajectory_id=trajectory_id,
                    step_number=step_number,
                )

                output_file.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                total_records += 1

            print(
                f"[{trajectory_number:3d}/"
                f"{TOTAL_TRAJECTORIES}] "
                f"trajectory generated "
                f"({total_records} records)"
            )

    expected_records = (
        TOTAL_TRAJECTORIES
        * STEPS_PER_TRAJECTORY
    )

    if total_records != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} records, "
            f"generated {total_records}."
        )

    print()
    print("=" * 70)
    print("AMI SECURITY DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(
        f"Trajectories: {TOTAL_TRAJECTORIES}"
    )
    print(
        f"Steps per trajectory: {STEPS_PER_TRAJECTORY}"
    )
    print(
        f"Total records: {total_records}"
    )
    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()