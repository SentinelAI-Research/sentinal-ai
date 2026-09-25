from backend.trajectory.models import TrajectoryStep


class TrajectoryStore:
    """
    In-memory store for agent trajectory steps.

    Steps are kept in execution order.
    """

    def __init__(self):
        self._steps: list[TrajectoryStep] = []

    def add_step(self, step: TrajectoryStep) -> None:
        """
        Add a trajectory step.

        Each step_id must be unique within the store.
        """

        if any(existing.step_id == step.step_id for existing in self._steps):
            raise ValueError(
                f"Trajectory step already exists: {step.step_id}"
            )

        self._steps.append(step)

    def get_steps(self) -> list[TrajectoryStep]:
        """Return all recorded steps in execution order."""
        return list(self._steps)

    def get_step(self, step_id: str) -> TrajectoryStep:
        """Return a specific trajectory step by ID."""
        for step in self._steps:
            if step.step_id == step_id:
                return step

        raise ValueError(f"Trajectory step not found: {step_id}")

    def count(self) -> int:
        """Return the number of recorded steps."""
        return len(self._steps)

    def clear(self) -> None:
        """Remove all recorded steps."""
        self._steps.clear()

    def get_agent_steps(self, agent_id: str) -> list[TrajectoryStep]:
        """
        Return all trajectory steps belonging to a specific agent,
        preserving execution order.
        """

        return [
            step
            for step in self._steps
            if step.agent_id == agent_id
        ]

    def get_session_steps(self, session_id: str) -> list[TrajectoryStep]:
        """
        Return all trajectory steps belonging to a specific session of an agent,
        preserving execution order.
        """

        return [
            step
            for step in self._steps
            if step.session_id == session_id
        ]

    def get_history(
        self,
        session_id: str,
        before_step_id: str,
    ) -> list[TrajectoryStep]:
        """
        Return all steps in a session that occurred before
        the specified step.
        """

        session_steps = self.get_session_steps(session_id)

        for index, step in enumerate(session_steps):
            if step.step_id == before_step_id:
                return session_steps[:index]

        raise ValueError(
            f"Step {before_step_id} not found in session {session_id}"
        )

    def latest_step(self) -> TrajectoryStep:
        """
        Return the most recently recorded trajectory step.
        """

        if not self._steps:
            raise ValueError("Trajectory store is empty.")

        return self._steps[-1]

    def session_count(self, session_id: str) -> int:
        """
        Return the number of recorded steps in a session.
        """

        return len(self.get_session_steps(session_id))