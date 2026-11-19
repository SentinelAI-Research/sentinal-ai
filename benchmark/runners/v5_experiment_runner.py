from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.agent.proposal_builder import ProposalBuilder
from backend.dataobjects.registry import DataObjectRegistry
from backend.monitor.config import MonitorConfig
from backend.monitor.factory import create_sentinel_runtime
from backend.schemas.contracts import Decision

DEFAULT_POLICY_PATH = ROOT / "policies" / "default.yaml"
V5_DIR = ROOT / "data" / "benchmark" / "v5"


def run_v5_scenario(
    scenario: dict[str, Any],
    monitor_config: MonitorConfig,
    *,
    policy_path: Path = DEFAULT_POLICY_PATH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Executes a single V5 scenario through the real SentinelRuntime under `monitor_config` (B3):
    - WARN is logged and execution continues (so early warnings can lead up to BLOCK or completion).
    - BLOCK stops the run immediately (action does not execute).
    - C0 (unmonitored) executes every step without stopping.
    Returns (step_records, trajectory_summary).
    """
    with tempfile.TemporaryDirectory(
        prefix=f"v5-{scenario['scenario_id']}-"
    ) as tmp_dir:
        workspace = Path(tmp_dir)
        runtime = create_sentinel_runtime(
            policy_path=str(policy_path),
            file_root=workspace,
            memory_path=workspace / "memory.json",
            monitor_config=monitor_config,
        )
        registry = DataObjectRegistry(runtime.provenance_graph)
        ref_to_object_id: dict[str, str] = {}
        ref_to_content: dict[str, str] = {}

        # 1. Write source files to workspace and register root DataObjects
        for src in scenario["source_files"]:
            file_path = workspace / f"{src['doc_id']}.txt"
            file_path.write_text(src["content"], encoding="utf-8")
            obj_id = f"{scenario['scenario_id']}-{src['ref']}"
            registry.register_source(
                object_id=obj_id,
                source=str(file_path),
                content=src["content"],
                sensitivity=src["sensitivity"],
                created_step=f"{scenario['scenario_id']}-init",
            )
            ref_to_object_id[src["ref"]] = obj_id
            ref_to_content[src["ref"]] = str(src["content"])

        step_records: list[dict[str, Any]] = []
        first_alert_step: int | None = None
        attack_executed = False
        completed = True
        builder = ProposalBuilder(agent_id=f"v5-agent-{scenario['scenario_id']}")

        # 2. Execute trajectory step by step
        for step in scenario["steps"]:
            step_num = int(step["step"])
            tool_name = str(step["tool"])
            destination = str(step.get("destination", "INTERNAL"))
            step_id = f"{scenario['scenario_id']}-step-{step_num:02d}"

            input_refs = list(step["input_refs"])
            input_obj_ids = [ref_to_object_id[ref] for ref in input_refs]
            combined_text = "\n".join(ref_to_content.get(ref, "") for ref in input_refs)
            raw_args = dict(step.get("arguments") or {})

            # Populate tool arguments cleanly according to ToolRegistry signatures
            if tool_name == "read_file":
                raw_args["path"] = str(workspace / raw_args["path"])
            elif tool_name in {"summarize", "paraphrase"}:
                raw_args["text"] = combined_text
            elif tool_name == "write_memory":
                raw_args = {"content": combined_text}
            elif tool_name == "read_memory":
                raw_args = {}
            elif tool_name == "send_message":
                raw_args["message"] = combined_text

            proposal = builder.build(
                step_id=step_id,
                tool_name=tool_name,
                arguments=raw_args,
                input_data_objects=input_obj_ids,
            )

            # Time the monitor evaluation call precisely (B3.2 & R4.3)
            t0 = time.perf_counter()
            decision = runtime.monitor.evaluate(proposal, destination)
            monitor_time_ms = round((time.perf_counter() - t0) * 1000.0, 4)

            prov = dict(getattr(runtime.monitor, "last_provenance_features", {}) or {})
            step_records.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "family": scenario["family"],
                    "label": scenario["label"],
                    "depth": scenario["depth"],
                    "config": monitor_config.name,
                    "step": step_num,
                    "tool": tool_name,
                    "destination": destination,
                    "verdict": decision.verdict,
                    "risk": decision.risk_score,
                    "policy_hits": "|".join(decision.policy_hits),
                    "has_sensitive_ancestor": prov.get("has_sensitive_ancestor", 0),
                    "sensitive_ancestor_count": prov.get("sensitive_ancestor_count", 0),
                    "provenance_depth": prov.get("provenance_depth", 0),
                    "transformation_depth": prov.get("transformation_depth", 0),
                    "num_parents": prov.get("num_parents", len(input_obj_ids)),
                    "monitor_time_ms": monitor_time_ms,
                }
            )

            if decision.verdict in {"WARN", "BLOCK"} and first_alert_step is None:
                first_alert_step = step_num

            # B3.3 Execution rule:
            # - BLOCK stops the run immediately
            # - WARN logs and continues execution
            if decision.verdict == "BLOCK":
                runtime.trajectory_store.update_decision(
                    step_id=step_id,
                    verdict="BLOCK",
                    risk_score=decision.risk_score,
                )
                completed = False
                break

            exec_decision = (
                Decision(
                    step_id=decision.step_id,
                    verdict="ALLOW",
                    risk_score=decision.risk_score,
                    risk_probability=decision.risk_probability,
                    policy_hits=decision.policy_hits,
                    provenance_trace=decision.provenance_trace,
                    reason=decision.reason,
                )
                if decision.verdict == "WARN"
                else decision
            )

            exec_result = runtime.runtime_executor.apply_decision(
                step_id=step_id,
                decision=exec_decision,
                tool_name=tool_name,
                arguments=raw_args,
            )
            if decision.verdict == "WARN":
                runtime.trajectory_store.update_decision(
                    step_id=step_id,
                    verdict="WARN",
                    risk_score=decision.risk_score,
                )

            if not exec_result.executed:
                completed = False
                break

            output_obj_id = runtime.output_handler.create_output_object(
                proposal=proposal,
                execution_result=exec_result,
            )
            exec_result.output_data_object_id = output_obj_id
            out_key = f"step_{step_num}_out"
            ref_to_object_id[out_key] = output_obj_id
            ref_to_content[out_key] = str(exec_result.output or combined_text)

            if scenario["label"] == "ATTACK" and step_num == scenario["violation_step"]:
                attack_executed = True

        trajectory_summary = {
            "scenario_id": scenario["scenario_id"],
            "family": scenario["family"],
            "label": scenario["label"],
            "depth": scenario["depth"],
            "config": monitor_config.name,
            "violation_step": scenario["violation_step"],
            "first_alert_step": first_alert_step,
            "attack_executed": attack_executed,
            "completed": completed,
        }
        return step_records, trajectory_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a V5 scenario under any MonitorConfig (M2)."
    )
    parser.add_argument("--partition", choices=["dev", "test"], default="dev")
    parser.add_argument("--family", default="memory_mediated_leakage")
    parser.add_argument(
        "--configs", nargs="+", default=["C0", "C1", "C2", "C2b", "C3", "C4"]
    )
    args = parser.parse_args()

    scenarios = json.loads(
        (V5_DIR / f"{args.partition}_scenarios.json").read_text(encoding="utf-8")
    )
    target = next((s for s in scenarios if s["family"] == args.family), scenarios[0])

    print(
        f"Running Scenario: {target['scenario_id']} | Family: {target['family']} | Label: {target['label']}"
    )
    for cfg_name in args.configs:
        cfg = MonitorConfig.from_preset(cfg_name)
        steps, summary = run_v5_scenario(target, cfg)
        print(f"\n--- Config {cfg_name} ---")
        print("Summary:", json.dumps(summary))
        for st in steps:
            print(
                f"  Step {st['step']:02d} | {st['tool']:<13} | dest={st['destination']:<8} "
                f"| verdict={st['verdict']:<5} | risk={st['risk']:.3f} "
                f"| sens_anc={st['has_sensitive_ancestor']} | t_depth={st['transformation_depth']} "
                f"| {st['monitor_time_ms']:.3f}ms"
            )


if __name__ == "__main__":
    main()
