from backend.policy.engine import PolicyEngine
from backend.risk.heuristic import HeuristicRiskScorer
from backend.provenance.graph import ProvenanceGraph
from backend.trajectory.store import TrajectoryStore
from backend.trajectory.models import TrajectoryStep
from backend.schemas.contracts import ToolCallProposal, Decision


class SentinelMonitor:
    """
    Central SentinelAI security monitor.

    Flow:

        ToolCallProposal
                ↓
        Trajectory Store
                ↓
        DataObject + Provenance
                ↓
        Policy Evaluation
                ↓
        Provenance Analysis
                ↓
        Risk Assessment
                ↓
        Security Decision

    The monitor evaluates the proposed action.
    Actual tool execution is handled separately by RuntimeExecutor.
    """

    def __init__(
        self,
        policy_engine: PolicyEngine,
        risk_scorer: HeuristicRiskScorer,
        provenance_graph: ProvenanceGraph,
        trajectory_store: TrajectoryStore,
    ):
        self.policy_engine = policy_engine
        self.risk_scorer = risk_scorer
        self.provenance_graph = provenance_graph
        self.trajectory_store = trajectory_store

    def evaluate(
        self,
        proposal: ToolCallProposal,
        destination: str,
    ) -> Decision:
        """
        Evaluate a tool-call proposal and return a security decision.
        """

        # ---------------------------------------------------------
        # 1. Validate input data objects
        # ---------------------------------------------------------

        if not proposal.input_data_objects:
            raise ValueError(f"Proposal {proposal.step_id} has no input data objects.")

        # ---------------------------------------------------------
        # 2. Determine the session ID
        #
        # ToolCallProposal currently does not contain session_id.
        # For now, use a deterministic session identifier based on
        # the agent.
        # ---------------------------------------------------------

        session_id = f"session-{proposal.agent_id}"

        # ---------------------------------------------------------
        # 3. Convert the proposal into a trajectory step
        # ---------------------------------------------------------

        step = TrajectoryStep(
            step_id=proposal.step_id,
            agent_id=proposal.agent_id,
            session_id=session_id,
            tool_name=proposal.tool_name,
            arguments=proposal.arguments,
            input_data_objects=proposal.input_data_objects,
        )

        # Store the proposed action in the trajectory.
        self.trajectory_store.add_step(step)

        # ---------------------------------------------------------
        # 4. Identify the input data object
        # ---------------------------------------------------------

        input_object_id = proposal.input_data_objects[-1]

        data_object = self.provenance_graph.get_object(input_object_id)

        # ---------------------------------------------------------
        # 5. Evaluate security policy
        # ---------------------------------------------------------

        policy_result = self.policy_engine.evaluate_action(
            source_sensitivity=data_object.sensitivity,
            destination=destination,
        )

        # ---------------------------------------------------------
        # 6. Calculate provenance depth
        # ---------------------------------------------------------

        provenance_depth = self.provenance_graph.get_depth(input_object_id)

        # ---------------------------------------------------------
        # 7. Retrieve previous trajectory history
        # ---------------------------------------------------------

        history = self.trajectory_store.get_history(
            session_id,
            proposal.step_id,
        )

        # ---------------------------------------------------------
        # 8. Run the risk scorer
        # ---------------------------------------------------------

        risk_result = self.risk_scorer.assess(
            policy_result=policy_result,
            history=history,
            provenance_depth=provenance_depth,
            destination=destination,
        )

        # ---------------------------------------------------------
        # 9. Build provenance trace
        # ---------------------------------------------------------

        ancestors = self.provenance_graph.get_ancestors(input_object_id)

        provenance_trace = [
            input_object_id,
            *ancestors,
        ]

        # ---------------------------------------------------------
        # 10. Create final security decision
        # ---------------------------------------------------------

        decision = Decision(
            step_id=proposal.step_id,
            verdict=policy_result.verdict,
            risk_score=risk_result.risk_score,
            risk_probability=None,
            policy_hits=policy_result.matched_rules,
            provenance_trace=provenance_trace,
            reason=risk_result.reason,
        )

        return decision
