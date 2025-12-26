"""Experiment storage for reproducibility."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from invariant.models.schemas import EvaluationResult, Hypothesis, RunRecord


class ExperimentStore:
    """Persistent storage for experiment runs and artifacts."""

    def __init__(self, base_path: Path | str = ".invariant"):
        self.base_path = Path(base_path)
        self.runs_path = self.base_path / "runs"
        self.hypotheses_path = self.base_path / "hypotheses"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.runs_path.mkdir(parents=True, exist_ok=True)
        self.hypotheses_path.mkdir(parents=True, exist_ok=True)

    def save_hypothesis(self, hypothesis: Hypothesis) -> Path:
        """Save a hypothesis to disk."""
        path = self.hypotheses_path / f"{hypothesis.id}.json"
        path.write_text(hypothesis.model_dump_json(indent=2))
        return path

    def load_hypothesis(self, hypothesis_id: UUID) -> Hypothesis:
        """Load a hypothesis from disk."""
        path = self.hypotheses_path / f"{hypothesis_id}.json"
        return Hypothesis.model_validate_json(path.read_text())

    def save_run(self, run: RunRecord) -> Path:
        """Save a run record."""
        date_prefix = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = self.runs_path / f"{date_prefix}_{run.id}.json"
        path.write_text(run.model_dump_json(indent=2))
        return path

    def load_run(self, run_id: UUID) -> RunRecord:
        """Load a run record."""
        # Search for the run file
        for path in self.runs_path.glob(f"*_{run_id}.json"):
            return RunRecord.model_validate_json(path.read_text())
        raise FileNotFoundError(f"Run {run_id} not found")

    def list_runs(
        self,
        hypothesis_id: UUID | None = None,
        limit: int = 100,
    ) -> list[RunRecord]:
        """List run records, optionally filtered by hypothesis."""
        runs = []
        for path in sorted(self.runs_path.glob("*.json"), reverse=True)[:limit]:
            run = RunRecord.model_validate_json(path.read_text())
            if hypothesis_id is None or run.hypothesis_id == hypothesis_id:
                runs.append(run)
        return runs

    def save_result(self, result: EvaluationResult) -> Path:
        """Save an evaluation result."""
        path = self.runs_path / f"result_{result.run_id}.json"
        path.write_text(result.model_dump_json(indent=2))
        return path

    def get_reproducibility_info(self, run_id: UUID) -> dict:
        """Get all info needed to reproduce a run."""
        run = self.load_run(run_id)
        hypothesis = self.load_hypothesis(run.hypothesis_id)
        return {
            "run": run.model_dump(),
            "hypothesis": hypothesis.model_dump(),
            "command": f"invariant run --seed {run.seed} --config '{json.dumps(run.config)}'",
        }
