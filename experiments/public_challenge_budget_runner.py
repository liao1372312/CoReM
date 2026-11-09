"""Budget-study runner that records selected-plan incompatibility as failure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from experiments.public_challenge_runner import PublicChallengeRunner


class PublicChallengeBudgetRunner(PublicChallengeRunner):
    def execute(
        self,
        event: Mapping[str, Any],
        memories: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        try:
            return self.base_execute(event, memories)
        except ValueError as error:
            if str(error) != "workflow step requires an unavailable binding":
                raise
            detail = str(error)
        except KeyError as error:
            # A hard-negative capture can request a field absent from the
            # observed response. This is a selected-plan failure, not a runner
            # infrastructure failure.
            detail = f"capture field unavailable: {error}"
        return {
            "completed": False,
            "failure": "selected plan is incompatible with observed runtime bindings",
            "failure_detail": detail,
            "selected_memory_observations": self._memory_observations(memories),
            "steps": [],
            "bindings": {},
            "communication": {
                "explicit_statuses": False,
                "outcome_summary": "Workflow could not establish its runtime handoffs.",
            },
        }

    def base_execute(
        self,
        event: Mapping[str, Any],
        memories: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Named seam used by tests without broadening the caught conditions."""
        return super().execute(event, memories)


def create_runner() -> PublicChallengeBudgetRunner:
    return PublicChallengeBudgetRunner()
