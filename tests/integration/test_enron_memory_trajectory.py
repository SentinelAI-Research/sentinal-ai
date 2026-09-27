import tarfile
from pathlib import Path

from backend.provenance.graph import ProvenanceGraph
from ml.data.enron_memory_trajectory import (
    EnronMemoryTrajectoryService,
)


ARCHIVE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "enron"
    / "enron_mail_20150507.tar.gz"
)


def _first_email_member() -> str:
    with tarfile.open(ARCHIVE_PATH, mode="r:gz") as tar:

        for member in tar:

            if not member.isfile():
                continue

            normalized = (
                member.name
                .replace("\\", "/")
                .strip("/")
            )

            parts = normalized.split("/")

            if len(parts) < 4:
                continue

            if parts[0].lower() != "maildir":
                continue

            if not parts[-1]:
                continue

            return member.name

    raise AssertionError(
        "No Enron email member found in archive."
    )


def test_enron_memory_mediated_trajectory():

    member_path = _first_email_member()

    graph, trajectory = (
        EnronMemoryTrajectoryService
        .build_from_archive_member(
            archive_path=ARCHIVE_PATH,
            member_path=member_path,
        )
    )

    assert isinstance(
        graph,
        ProvenanceGraph,
    )

    assert len(trajectory) == 6

    assert len(graph.graph.nodes) == 6
    assert len(graph.graph.edges) == 5

    source_id = trajectory[0]
    summary_id = trajectory[1]
    memory_id = trajectory[2]
    retrieved_id = trajectory[3]
    paraphrase_id = trajectory[4]
    message_id = trajectory[5]

    # ---------------------------------------------------------
    # Parent relationships
    # ---------------------------------------------------------

    assert graph.get_parents(source_id) == []

    assert graph.get_parents(summary_id) == [
        source_id
    ]

    assert graph.get_parents(memory_id) == [
        summary_id
    ]

    assert graph.get_parents(retrieved_id) == [
        memory_id
    ]

    assert graph.get_parents(paraphrase_id) == [
        retrieved_id
    ]

    assert graph.get_parents(message_id) == [
        paraphrase_id
    ]

    # ---------------------------------------------------------
    # Graph depth
    # ---------------------------------------------------------

    assert graph.get_depth(source_id) == 0
    assert graph.get_depth(summary_id) == 1
    assert graph.get_depth(memory_id) == 2
    assert graph.get_depth(retrieved_id) == 3
    assert graph.get_depth(paraphrase_id) == 4
    assert graph.get_depth(message_id) == 5

    # ---------------------------------------------------------
    # Transformation depth
    # ---------------------------------------------------------

    assert graph.get_transformation_depth(source_id) == 1
    assert graph.get_transformation_depth(summary_id) == 2
    assert graph.get_transformation_depth(memory_id) == 3
    assert graph.get_transformation_depth(retrieved_id) == 4
    assert graph.get_transformation_depth(paraphrase_id) == 5
    assert graph.get_transformation_depth(message_id) == 6

    # ---------------------------------------------------------
    # Ordered provenance path
    # ---------------------------------------------------------

    assert graph.get_path(
        source_id,
        message_id,
    ) == [
        source_id,
        summary_id,
        memory_id,
        retrieved_id,
        paraphrase_id,
        message_id,
    ]

    # ---------------------------------------------------------
    # Ancestor membership
    # ---------------------------------------------------------

    ancestors = graph.get_ancestors(
        message_id
    )

    assert set(ancestors) == {
        source_id,
        summary_id,
        memory_id,
        retrieved_id,
        paraphrase_id,
    }