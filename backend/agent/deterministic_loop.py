from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from backend.agent.proposal_builder import ProposalBuilder
from backend.monitor.factory import SentinelRuntime
from backend.policy.models import POLICY_DESTINATIONS
from backend.schemas.contracts import ToolCallProposal

ExecutionStatus = Literal["not_attempted", "succeeded", "failed"]
TaskStatus = Literal["completed", "stopped", "failed"]


@dataclass(frozen=True)
class PlannedAction:
    """A deterministic proposal description; it cannot execute tools."""

    step_id: str
    tool_name: str
    destination: str
    description: str
    arguments: dict[str, Any] = field(default_factory=dict)
    input_data_objects: tuple[str, ...] = ()
    input_from_previous_output: bool = False
    previous_output_argument: str | None = None


@dataclass(frozen=True)
class AgentActionTrace:
    run_id: str
    action_index: int
    step_id: str
    description: str
    tool_name: str
    input_data_object_ids: list[str]
    destination: str
    verdict: str | None
    decision_reason: str | None
    matched_policy_rules: list[str]
    risk_score: float | None
    risk_probability: float | None
    execution_status: ExecutionStatus
    error: str | None
    output_data_object_ids: list[str]
    provenance_reference_ids: list[str]
    output_preview: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "action_index": self.action_index,
            "step_id": self.step_id,
            "description": self.description,
            "tool_name": self.tool_name,
            "input_data_object_ids": self.input_data_object_ids,
            "destination": self.destination,
            "verdict": self.verdict,
            "decision_reason": self.decision_reason,
            "matched_policy_rules": self.matched_policy_rules,
            "risk_score": self.risk_score,
            "risk_probability": self.risk_probability,
            "execution_status": self.execution_status,
            "error": self.error,
            "output_data_object_ids": self.output_data_object_ids,
            "provenance_reference_ids": self.provenance_reference_ids,
            "output_preview": self.output_preview,
        }


@dataclass(frozen=True)
class DeterministicTaskResult:
    run_id: str
    task_description: str
    status: TaskStatus
    stop_reason: str
    trace: list[AgentActionTrace]
    provenance_objects: list[dict[str, Any]]
    provenance_edges: list[dict[str, str]]
    planned_action_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_description": self.task_description,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "summary": {
                "planned_actions": self.planned_action_count,
                "proposed_actions": len(self.trace),
                "executed_actions": sum(
                    step.execution_status == "succeeded" for step in self.trace
                ),
                "warnings": sum(step.verdict == "WARN" for step in self.trace),
                "blocked_actions": sum(step.verdict == "BLOCK" for step in self.trace),
                "failed_executions": sum(
                    step.execution_status == "failed" for step in self.trace
                ),
            },
            "trace": [step.as_dict() for step in self.trace],
            "provenance": {
                "objects": self.provenance_objects,
                "edges": self.provenance_edges,
            },
        }


class DeterministicPlanner:
    """Returns a supplied plan in order and has no access to tool functions."""

    def propose(
        self,
        task_description: str,
        action_plan: list[PlannedAction],
    ) -> list[PlannedAction]:
        if not task_description.strip():
            raise ValueError("Task description cannot be empty.")
        if not action_plan:
            raise ValueError("Action plan cannot be empty.")
        return list(action_plan)


class DeterministicAgentLoop:
    def __init__(
        self,
        runtime: SentinelRuntime,
        agent_id: str,
        planner: DeterministicPlanner | None = None,
    ):
        self.runtime = runtime
        self.agent_id = agent_id
        self.planner = planner or DeterministicPlanner()
        self.proposal_builder = ProposalBuilder(agent_id=agent_id)

    def run(
        self,
        task_description: str,
        action_plan: list[PlannedAction],
        run_id: str | None = None,
    ) -> DeterministicTaskResult:
        planned_actions = self.planner.propose(task_description, action_plan)
        self._validate_plan(planned_actions)
        run_id = run_id or str(uuid4())

        trace: list[AgentActionTrace] = []
        referenced_objects: list[str] = []
        previous_output: str | None = None
        previous_output_object_id: str | None = None
        task_status: TaskStatus = "completed"
        stop_reason = "All planned actions executed successfully."

        for action_index, action in enumerate(planned_actions, start=1):
            input_ids = list(action.input_data_objects)
            if action.input_from_previous_output:
                if previous_output_object_id is None:
                    task_status = "failed"
                    stop_reason = "Previous action did not produce an input object."
                    break
                input_ids.append(previous_output_object_id)

            arguments = dict(action.arguments)
            if action.previous_output_argument is not None:
                if previous_output is None:
                    task_status = "failed"
                    stop_reason = "Previous action did not produce tool output."
                    break
                arguments[action.previous_output_argument] = previous_output

            referenced_objects.extend(input_ids)
            try:
                proposal: ToolCallProposal = self.proposal_builder.build(
                    step_id=action.step_id,
                    tool_name=action.tool_name,
                    arguments=arguments,
                    input_data_objects=input_ids,
                )
                decision, execution_result = self.runtime.monitor_service.process(
                    proposal=proposal,
                    destination=action.destination,
                )
            except Exception as exc:
                trace.append(
                    self._trace(
                        run_id=run_id,
                        action_index=action_index,
                        action=action,
                        input_ids=input_ids,
                        execution_status="not_attempted",
                        error=str(exc),
                    )
                )
                task_status = "failed"
                stop_reason = f"Action {action.step_id} failed before execution."
                break

            referenced_objects.extend(decision.provenance_trace)
            if decision.verdict != "ALLOW":
                trace.append(
                    self._trace(
                        run_id=run_id,
                        action_index=action_index,
                        action=action,
                        input_ids=input_ids,
                        verdict=decision.verdict,
                        reason=decision.reason,
                        policy_hits=decision.policy_hits,
                        risk_score=decision.risk_score,
                        risk_probability=decision.risk_probability,
                        provenance_ids=decision.provenance_trace,
                    )
                )
                task_status = "stopped"
                stop_reason = f"Stopped by {decision.verdict}: {decision.reason}"
                break

            if not execution_result.executed:
                trace.append(
                    self._trace(
                        run_id=run_id,
                        action_index=action_index,
                        action=action,
                        input_ids=input_ids,
                        verdict=decision.verdict,
                        reason=decision.reason,
                        policy_hits=decision.policy_hits,
                        risk_score=decision.risk_score,
                        risk_probability=decision.risk_probability,
                        execution_status="failed",
                        error=execution_result.error or "Tool execution failed.",
                        provenance_ids=decision.provenance_trace,
                    )
                )
                task_status = "failed"
                stop_reason = f"Tool {action.tool_name} failed during execution."
                break

            output_id = execution_result.output_data_object_id
            if not output_id:
                trace.append(
                    self._trace(
                        run_id=run_id,
                        action_index=action_index,
                        action=action,
                        input_ids=input_ids,
                        verdict=decision.verdict,
                        reason=decision.reason,
                        policy_hits=decision.policy_hits,
                        risk_score=decision.risk_score,
                        risk_probability=decision.risk_probability,
                        execution_status="succeeded",
                        error=(
                            execution_result.error
                            or "Tool executed but no output DataObject was recorded."
                        ),
                        provenance_ids=decision.provenance_trace,
                    )
                )
                task_status = "failed"
                stop_reason = "Tool output provenance could not be recorded."
                break

            output_object = self.runtime.provenance_graph.get_object(output_id)
            referenced_objects.append(output_id)
            trace.append(
                self._trace(
                    run_id=run_id,
                    action_index=action_index,
                    action=action,
                    input_ids=input_ids,
                    verdict=decision.verdict,
                    reason=decision.reason,
                    policy_hits=decision.policy_hits,
                    risk_score=decision.risk_score,
                    risk_probability=decision.risk_probability,
                    execution_status="succeeded",
                    output_ids=[output_id],
                    provenance_ids=decision.provenance_trace + [output_id],
                    output_preview=(
                        execution_result.output[:240]
                        if output_object.sensitivity == "PUBLIC"
                        and execution_result.output is not None
                        else None
                    ),
                )
            )
            previous_output = execution_result.output
            previous_output_object_id = output_id

        provenance_objects, provenance_edges = self._provenance_snapshot(
            referenced_objects
        )
        return DeterministicTaskResult(
            run_id=run_id,
            task_description=task_description,
            status=task_status,
            stop_reason=stop_reason,
            trace=trace,
            provenance_objects=provenance_objects,
            provenance_edges=provenance_edges,
            planned_action_count=len(planned_actions),
        )

    def _validate_plan(self, actions: list[PlannedAction]) -> None:
        existing_step_ids = {
            step.step_id for step in self.runtime.trajectory_store.get_steps()
        }
        seen_step_ids: set[str] = set()
        for index, action in enumerate(actions):
            if not action.step_id.strip() or action.step_id in seen_step_ids:
                raise ValueError("Plan step IDs must be non-empty and unique.")
            if action.step_id in existing_step_ids:
                raise ValueError(f"Plan step ID already exists: {action.step_id}")
            if not action.tool_name.strip() or not self.runtime.tool_executor.has_tool(
                action.tool_name
            ):
                raise ValueError(f"Unsupported tool: {action.tool_name}")
            if action.destination not in POLICY_DESTINATIONS:
                raise ValueError(f"Unsupported destination: {action.destination}")
            if not isinstance(action.arguments, dict):
                raise ValueError(f"Arguments for {action.step_id} must be a mapping.")
            if (
                action.previous_output_argument
                and not action.input_from_previous_output
            ):
                raise ValueError(
                    "Previous-output arguments require the previous output as an input."
                )
            if action.input_from_previous_output and index == 0:
                raise ValueError("The first action cannot reference a previous output.")
            if not action.input_data_objects and not action.input_from_previous_output:
                raise ValueError(
                    f"Action {action.step_id} requires a valid provenance input."
                )
            if action.input_data_objects:
                self.runtime.provenance_graph.resolve_objects(
                    list(action.input_data_objects)
                )
            if action.input_from_previous_output:
                previous_output_id = f"{actions[index - 1].step_id}-output"
                if previous_output_id in action.input_data_objects:
                    raise ValueError(
                        "Plan contains a duplicate previous-output parent."
                    )
            if self.runtime.provenance_graph.has_object(f"{action.step_id}-output"):
                raise ValueError(
                    f"Output object ID already exists for {action.step_id}."
                )
            seen_step_ids.add(action.step_id)

    def _trace(
        self,
        run_id: str,
        action_index: int,
        action: PlannedAction,
        input_ids: list[str],
        verdict: str | None = None,
        reason: str | None = None,
        policy_hits: list[str] | None = None,
        risk_score: float | None = None,
        risk_probability: float | None = None,
        execution_status: ExecutionStatus = "not_attempted",
        error: str | None = None,
        output_ids: list[str] | None = None,
        provenance_ids: list[str] | None = None,
        output_preview: str | None = None,
    ) -> AgentActionTrace:
        return AgentActionTrace(
            run_id=run_id,
            action_index=action_index,
            step_id=action.step_id,
            description=action.description,
            tool_name=action.tool_name,
            input_data_object_ids=input_ids,
            destination=action.destination,
            verdict=verdict,
            decision_reason=reason,
            matched_policy_rules=policy_hits or [],
            risk_score=risk_score,
            risk_probability=risk_probability,
            execution_status=execution_status,
            error=error,
            output_data_object_ids=output_ids or [],
            provenance_reference_ids=list(dict.fromkeys(provenance_ids or input_ids)),
            output_preview=output_preview,
        )

    def _provenance_snapshot(
        self,
        object_ids: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        graph = self.runtime.provenance_graph
        ordered_ids: list[str] = []
        for object_id in object_ids:
            if not graph.has_object(object_id):
                continue
            for ancestor_id in sorted(graph.get_ancestors(object_id)):
                if ancestor_id not in ordered_ids:
                    ordered_ids.append(ancestor_id)
            if object_id not in ordered_ids:
                ordered_ids.append(object_id)

        objects = []
        edges: set[tuple[str, str]] = set()
        for object_id in ordered_ids:
            data_object = graph.get_object(object_id)
            parents = sorted(graph.get_parents(object_id))
            objects.append(
                {
                    "object_id": object_id,
                    "sensitivity": data_object.sensitivity,
                    "transformation": data_object.transformation,
                    "parents": parents,
                }
            )
            edges.update((parent_id, object_id) for parent_id in parents)

        order = {object_id: index for index, object_id in enumerate(ordered_ids)}
        return objects, [
            {"parent_id": parent_id, "child_id": child_id}
            for parent_id, child_id in sorted(
                edges,
                key=lambda edge: (order[edge[1]], order[edge[0]]),
            )
        ]
