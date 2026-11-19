from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.dataobjects import sensitivity as sensitivity_mod
from backend.dataobjects.models import DataObject
from backend.dataobjects.registry import DataObjectRegistry
from backend.provenance.graph import ProvenanceGraph

V5_DIR = ROOT / "data" / "benchmark" / "v5"

SENSITIVITY_ORDER = {
    "PUBLIC": 0,
    "INTERNAL": 1,
    "CONFIDENTIAL": 2,
}


def _inherit_sensitivity(parent_sensitivities: list[str]) -> str:
    """Uses the project's sensitivity resolver if exported, otherwise standard dominance."""
    for fn_name in (
        "resolve_inherited_sensitivity",
        "inherit_sensitivity",
        "max_sensitivity",
        "combine_sensitivities",
    ):
        fn = getattr(sensitivity_mod, fn_name, None)
        if callable(fn):
            return fn(parent_sensitivities)

    if not parent_sensitivities:
        return "PUBLIC"
    return max(
        parent_sensitivities,
        key=lambda level: SENSITIVITY_ORDER.get(str(level).upper(), 0),
    )


def _register_derived_object(
    registry: DataObjectRegistry,
    graph: ProvenanceGraph,
    *,
    object_id: str,
    source: str,
    content: str,
    sensitivity: str,
    parents: list[str],
    transformation: str,
    created_step: str,
) -> None:
    """Registers a derived DataObject and its parent lineage edges in ProvenanceGraph."""
    for method_name in (
        "register_derived",
        "register_output",
        "register_transformation",
    ):
        method = getattr(registry, method_name, None)
        if callable(method):
            sig = inspect.signature(method)
            kwargs = {
                "object_id": object_id,
                "source": source,
                "content": content,
                "sensitivity": sensitivity,
                "parents": parents,
                "parent_ids": parents,
                "transformation": transformation,
                "created_step": created_step,
            }
            filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
            method(**filtered)
            return

    # Direct DataObject + ProvenanceGraph registration matching backend/dataobjects/models.py
    obj_sig = inspect.signature(DataObject)
    obj_kwargs = {
        "object_id": object_id,
        "source": source,
        "content": content,
        "sensitivity": sensitivity,
        "parents": parents,
        "transformation": transformation,
        "created_step": created_step,
    }
    data_obj = DataObject(
        **{k: v for k, v in obj_kwargs.items() if k in obj_sig.parameters}
    )

    if hasattr(graph, "add_object"):
        graph.add_object(data_obj)
    elif hasattr(graph, "register_object"):
        graph.register_object(data_obj)
    else:
        raise RuntimeError("ProvenanceGraph has no object registration method.")

    for parent_id in parents:
        if hasattr(graph, "add_edge"):
            graph.add_edge(parent_id, object_id)
        elif hasattr(graph, "link"):
            graph.link(parent_id, object_id)


def verify_scenario_ground_truth(scenario: dict) -> tuple[bool, str]:
    """
    Executes the scenario's DAG lineage under C0 (no monitor) using the real
    ProvenanceGraph and sensitivity inheritance engine, then verifies (B2.2):
    - Every ATTACK scenario's sent object has a CONFIDENTIAL ancestor.
    - No BENIGN scenario's sent object has a CONFIDENTIAL ancestor.
    """
    graph = ProvenanceGraph()
    registry = DataObjectRegistry(graph)
    ref_to_object_id: dict[str, str] = {}

    # 1. Register root source DataObjects
    for src in scenario["source_files"]:
        obj_id = f"{scenario['scenario_id']}-{src['ref']}"
        registry.register_source(
            object_id=obj_id,
            source=src["doc_id"],
            content=src["content"],
            sensitivity=src["sensitivity"],
            created_step=f"{scenario['scenario_id']}-init",
        )
        ref_to_object_id[src["ref"]] = obj_id

    # 2. Replay every step in the DAG (C0: no monitor blocks execution)
    sent_object_ids: list[str] = []
    for step in scenario["steps"]:
        step_num = step["step"]
        tool = step["tool"]
        parent_ids = [ref_to_object_id[ref] for ref in step["input_refs"]]
        parent_objs = [graph.get_object(pid) for pid in parent_ids]
        inherited_sens = _inherit_sensitivity([p.sensitivity for p in parent_objs])

        out_id = f"{scenario['scenario_id']}-step_{step_num}_out"
        _register_derived_object(
            registry,
            graph,
            object_id=out_id,
            source=f"tool:{tool}",
            content=f"derived_output_{step_num}",
            sensitivity=inherited_sens,
            parents=parent_ids,
            transformation=tool,
            created_step=f"{scenario['scenario_id']}-step-{step_num}",
        )
        ref_to_object_id[f"step_{step_num}_out"] = out_id

        if tool == "send_message" and step.get("destination") == "EXTERNAL":
            sent_object_ids.extend(parent_ids)

    # 3. Inspect provenance ancestors of all externally sent objects
    has_confidential_outbound_ancestor = False
    for obj_id in sent_object_ids:
        lineage_ids = [obj_id] + list(graph.get_ancestors(obj_id))
        if any(
            graph.get_object(lid).sensitivity == "CONFIDENTIAL" for lid in lineage_ids
        ):
            has_confidential_outbound_ancestor = True
            break

    label = scenario["label"]
    if label == "ATTACK" and not has_confidential_outbound_ancestor:
        return (
            False,
            f"{scenario['scenario_id']}: ATTACK sent object lacks CONFIDENTIAL ancestor.",
        )
    if label == "BENIGN" and has_confidential_outbound_ancestor:
        return (
            False,
            f"{scenario['scenario_id']}: BENIGN sent object leaked CONFIDENTIAL ancestor.",
        )
    return True, "OK"


def validate_v5_directory(v5_dir: Path = V5_DIR) -> dict[str, int]:
    dev_scenarios = json.loads(
        (v5_dir / "dev_scenarios.json").read_text(encoding="utf-8")
    )
    test_scenarios = json.loads(
        (v5_dir / "test_scenarios.json").read_text(encoding="utf-8")
    )
    all_scenarios = dev_scenarios + test_scenarios

    failures: list[str] = []
    for scenario in all_scenarios:
        ok, msg = verify_scenario_ground_truth(scenario)
        if not ok:
            failures.append(msg)

    if failures:
        raise RuntimeError(
            f"B2 Ground-Truth Sanity Check failed on {len(failures)} scenarios:\n"
            + "\n".join(failures[:10])
        )

    return {
        "total_verified": len(all_scenarios),
        "dev_verified": len(dev_scenarios),
        "test_verified": len(test_scenarios),
        "failures": 0,
    }


if __name__ == "__main__":
    summary = validate_v5_directory()
    print("B2 Ground-Truth Sanity Check PASSED:", json.dumps(summary, indent=2))
