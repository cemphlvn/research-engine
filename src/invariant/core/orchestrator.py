"""Orchestrator: Scientific method in code.

Architecture:
- LLM Provider = hypothesis generator + transformation generator + adversary
- Embedding Provider = measurement instrument
- Stats Engine = epistemic court
- Orchestrator = this module - wires them together as scientific method
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Any
from uuid import uuid4
from enum import Enum
import random


class EvalStep(str, Enum):
    """Steps in the evaluation pipeline."""
    HYPOTHESIS_GEN = "hypothesis_generation"
    SCOREBOARD_SETUP = "scoreboard_setup"
    NEGATIVE_GEN = "negative_generation"
    TRIAL_START = "trial_start"
    EMBED_ORIGINAL = "embed_original"
    APPLY_TRANSFORMS = "apply_transforms"
    EMBED_TRANSFORMED = "embed_transformed"
    TRIAL_END = "trial_end"
    COMPUTE_METRICS = "compute_metrics"
    CHECK_ADMISSIBILITY = "check_admissibility"
    APPLY_DECISION = "apply_decision"
    PACKAGE_RESULT = "package_result"


@dataclass
class StepEvent:
    """Event emitted during evaluation."""
    step: EvalStep
    status: str  # "start", "complete", "error"
    message: str
    data: dict = field(default_factory=dict)
    layer: str = ""  # "llm", "embedding", "stats", "orchestrator"


# Type for progress callback
ProgressCallback = Callable[[StepEvent], None]

import numpy as np

from invariant.models.schemas import (
    AdmissibilityResult,
    ArtifactRef,
    BoundaryCondition,
    ClaimTarget,
    ComparisonOp,
    DecisionPolicy,
    DiscoveryReport,
    EvaluationOutcome,
    EvaluationResult,
    ExperimentPlan,
    FalsificationReport,
    FalsificationRule,
    FailureMode,
    Hypothesis,
    HypothesisType,
    InvarianceScore,
    Lens,
    MetricComputation,
    MetricName,
    MetricValue,
    MinimalCounterexample,
    ModelReference,
    NegativeDataset,
    NegativeGeneratorConfig,
    NegativeStrategy,
    Provenance,
    RunRecord,
    ScoreboardSpec,
    ThresholdRule,
    Transform,
    TransformCategory,
    TransformSet,
    UnderdeterminedReport,
    AdmissibilityConfig,
    AdmissibilityRule,
    TrialPlan,
)
from invariant.providers.base import (
    LLMProvider,
    EmbeddingProvider,
    StubLLMProvider,
    StubEmbeddingProvider,
    ProviderConfig,
)
from invariant.providers.stats import StatsEngine, StatsConfig
from invariant.store.experiment import ExperimentStore


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    seed: int = 42
    max_iterations: int = 100
    convergence_threshold: float = 0.01
    budget_limit: int = 1000
    store_path: str = ".invariant"

    # Default thresholds
    stability_threshold: float = 0.80
    separability_threshold: float = 0.75
    mi_threshold: float = 0.50
    kl_max: float = 2.0  # Very relaxed for stub testing; tighten for production


@dataclass
class TrialResult:
    """Result of a single trial."""

    seed: int
    original_embeddings: np.ndarray
    transformed_embeddings: np.ndarray
    negative_embeddings: np.ndarray
    transforms_applied: list[str]
    error: str | None = None


class Orchestrator:
    """Main orchestrator - the scientific method in code.

    Wires together:
    - LLM Provider (hypothesis gen, transform gen, adversary)
    - Embedding Provider (measurement instrument)
    - Stats Engine (epistemic court)

    Implements the 6-step evaluation pipeline:
    1. Hypothesis generation (LLM)
    2. Transform application (LLM)
    3. Negative generation (LLM as adversary)
    4. Embedding (measurement)
    5. Metric computation + decision (stats court)
    6. Result packaging
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        llm: LLMProvider | None = None,
        embedding: EmbeddingProvider | None = None,
        stats: StatsEngine | None = None,
    ):
        self.config = config or OrchestratorConfig()
        self.store = ExperimentStore(self.config.store_path)

        # Initialize providers - prefer real providers when API keys available
        self.llm = llm or self._create_llm_provider()
        self.embedding = embedding or self._create_embedding_provider()
        self.stats = stats or StatsEngine(StatsConfig())

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _create_llm_provider(self) -> LLMProvider:
        """Create LLM provider based on available API keys."""
        import os
        provider = os.environ.get("INVARIANT_LLM_PROVIDER", "deepseek").lower()

        # Try providers based on explicit config or available keys
        if provider == "deepseek" or os.environ.get("DEEPSEEK_API_KEY"):
            try:
                from invariant.providers.semantic import create_semantic_provider_deepseek
                return create_semantic_provider_deepseek()
            except Exception:
                pass

        if provider == "openai" or os.environ.get("OPENAI_API_KEY"):
            try:
                from invariant.providers.semantic import create_semantic_provider_openai
                return create_semantic_provider_openai()
            except Exception:
                pass

        if provider == "anthropic" or os.environ.get("ANTHROPIC_API_KEY"):
            try:
                from invariant.providers.semantic import create_semantic_provider_anthropic
                return create_semantic_provider_anthropic()
            except Exception:
                pass

        # Fallback to stub
        stub_config = ProviderConfig(
            model_ref=ModelReference(provider="stub", model_id="stub-v1")
        )
        return StubLLMProvider(stub_config)

    def _create_embedding_provider(self) -> EmbeddingProvider:
        """Create embedding provider based on available API keys."""
        import os

        # Prefer OpenAI embeddings (best quality/price)
        if os.environ.get("OPENAI_API_KEY"):
            try:
                from invariant.providers.openai_provider import OpenAIEmbeddingProvider
                return OpenAIEmbeddingProvider()
            except Exception:
                pass

        # Fallback to stub
        stub_config = ProviderConfig(
            model_ref=ModelReference(provider="stub", model_id="stub-v1")
        )
        return StubEmbeddingProvider(stub_config)

    # =========================================================================
    # MAIN EVALUATION
    # =========================================================================

    def evaluate(
        self,
        claim: str,
        positives: list[str],
        hypothesis_type: HypothesisType = HypothesisType.CONCEPT_INVARIANT,
        scoreboard: ScoreboardSpec | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> EvaluationResult:
        """Evaluate a claim for invariance.

        This is the main entry point. Given a claim and positive examples,
        run the full scientific method:
        1. Generate hypothesis (LLM)
        2. Generate negatives (LLM as adversary)
        3. Apply transforms (LLM)
        4. Embed everything (measurement)
        5. Compute metrics + decide (stats court)
        6. Package result

        Args:
            claim: The claim to test
            positives: Positive examples supporting the claim
            hypothesis_type: Type of hypothesis
            scoreboard: Optional custom ScoreboardSpec (uses defaults if None)
            on_progress: Optional callback for progress events

        Returns:
            EvaluationResult with outcome and detailed report
        """
        def emit(step: EvalStep, status: str, msg: str, layer: str = "orchestrator", **data):
            if on_progress:
                on_progress(StepEvent(step=step, status=status, message=msg, layer=layer, data=data))

        # 1. Generate hypothesis (LLM layer)
        emit(EvalStep.HYPOTHESIS_GEN, "start", f"Generating hypothesis from: '{claim[:50]}...'", "llm")
        hypothesis = self.llm.generate_hypothesis(claim, hypothesis_type)
        emit(EvalStep.HYPOTHESIS_GEN, "complete", f"Hypothesis: {hypothesis.title}", "llm",
             predicted_invariants=hypothesis.predicted_invariants)

        # 2. Create or use scoreboard
        emit(EvalStep.SCOREBOARD_SETUP, "start", "Setting up scoreboard (rules of winning)", "orchestrator")
        spec = scoreboard or self._default_scoreboard(hypothesis)
        emit(EvalStep.SCOREBOARD_SETUP, "complete", f"Scoreboard: {spec.name}", "orchestrator",
             metrics=[m.name.value for m in spec.metrics],
             admissibility_rules=[r.name for r in spec.admissibility.must_hold])

        # 3. Generate negatives (LLM as adversary)
        emit(EvalStep.NEGATIVE_GEN, "start", f"Generating adversarial negatives from {len(positives)} positives", "llm")
        negatives = self.llm.generate_negatives(
            positives,
            NegativeStrategy.NEAR_MISS,
            n_per_positive=2,
        )
        emit(EvalStep.NEGATIVE_GEN, "complete", f"Generated {len(negatives)} near-miss negatives", "llm",
             examples=negatives[:3])

        # 4. Run trials
        trials = self._run_trials(spec, positives, negatives, on_progress)

        # 5. Compute metrics (stats court)
        emit(EvalStep.COMPUTE_METRICS, "start", "Computing invariance metrics (epistemic court)", "stats")
        if not trials:
            emit(EvalStep.COMPUTE_METRICS, "error", "No successful trials", "stats")
            return self._empty_result(hypothesis, "No successful trials")

        all_orig = np.vstack([t.original_embeddings for t in trials])
        all_trans = np.vstack([t.transformed_embeddings for t in trials])
        all_neg = np.vstack([t.negative_embeddings for t in trials])

        metrics = self.stats.compute_metrics(
            all_orig, all_trans, all_neg, spec.metrics
        )
        metrics_dict = metrics.to_dict()
        emit(EvalStep.COMPUTE_METRICS, "complete", "Metrics computed", "stats",
             metrics={k.value: f"{v:.3f}" for k, v in metrics_dict.items()})

        # 6. Check admissibility
        emit(EvalStep.CHECK_ADMISSIBILITY, "start", "Checking transform admissibility", "stats")
        admissibility = self.stats.check_admissibility(spec.admissibility, metrics_dict)
        if not admissibility.admissible:
            emit(EvalStep.CHECK_ADMISSIBILITY, "complete", f"INADMISSIBLE: {admissibility.failing_rules}", "stats",
                 failing=admissibility.failing_rules)
            return self._inadmissible_result(hypothesis, admissibility)
        emit(EvalStep.CHECK_ADMISSIBILITY, "complete", "All admissibility rules passed", "stats")

        # 7. Apply decision logic
        emit(EvalStep.APPLY_DECISION, "start", "Applying decision policy", "stats")
        outcome, failure_mode = self.stats.apply_decision(spec.decision, metrics_dict)
        emit(EvalStep.APPLY_DECISION, "complete", f"Decision: {outcome.value.upper()}", "stats",
             outcome=outcome.value, failure_mode=failure_mode.value if failure_mode else None)

        # 8. Package result
        emit(EvalStep.PACKAGE_RESULT, "start", "Packaging final result", "orchestrator")
        result = self._package_result(
            hypothesis, spec, metrics, admissibility, outcome, failure_mode, trials
        )
        emit(EvalStep.PACKAGE_RESULT, "complete", f"Evaluation complete: {outcome.value}", "orchestrator")

        return result

    def evaluate_with_scoreboard(
        self,
        hypothesis: Hypothesis,
        scoreboard: ScoreboardSpec,
        positives: list[str],
    ) -> EvaluationResult:
        """Evaluate using explicit hypothesis and scoreboard."""
        # Generate negatives
        negatives = self.llm.generate_negatives(
            positives,
            NegativeStrategy.NEAR_MISS,
            n_per_positive=2,
        )

        # Run trials
        trials = self._run_trials(scoreboard, positives, negatives)

        if not trials:
            return self._empty_result(hypothesis, "No successful trials")

        # Compute metrics
        all_orig = np.vstack([t.original_embeddings for t in trials])
        all_trans = np.vstack([t.transformed_embeddings for t in trials])
        all_neg = np.vstack([t.negative_embeddings for t in trials])

        metrics = self.stats.compute_metrics(
            all_orig, all_trans, all_neg, scoreboard.metrics
        )
        metrics_dict = metrics.to_dict()

        # Check admissibility
        admissibility = self.stats.check_admissibility(
            scoreboard.admissibility, metrics_dict
        )

        if not admissibility.admissible:
            return self._inadmissible_result(hypothesis, admissibility)

        # Decide
        outcome, failure_mode = self.stats.apply_decision(
            scoreboard.decision, metrics_dict
        )

        return self._package_result(
            hypothesis, scoreboard, metrics, admissibility, outcome, failure_mode, trials
        )

    # =========================================================================
    # TRIAL EXECUTION
    # =========================================================================

    def _run_trials(
        self,
        spec: ScoreboardSpec,
        positives: list[str],
        negatives: list[str],
        on_progress: ProgressCallback | None = None,
    ) -> list[TrialResult]:
        """Run all trials according to scoreboard spec."""
        def emit(step: EvalStep, status: str, msg: str, layer: str = "orchestrator", **data):
            if on_progress:
                on_progress(StepEvent(step=step, status=status, message=msg, layer=layer, data=data))

        results = []
        transform_set = self._resolve_transform_set(spec.transform_set)
        total_trials = len(spec.trial_plan.seeds)

        for i, seed in enumerate(spec.trial_plan.seeds):
            random.seed(seed)
            np.random.seed(seed)

            emit(EvalStep.TRIAL_START, "start", f"Trial {i+1}/{total_trials} (seed={seed})", "orchestrator",
                 trial=i+1, total=total_trials, seed=seed)

            try:
                # Embed originals (measurement)
                emit(EvalStep.EMBED_ORIGINAL, "start", f"Embedding {len(positives)} positives + {len(negatives)} negatives", "embedding")
                original_emb = self.embedding.embed(positives)
                negative_emb = self.embedding.embed(negatives)
                emit(EvalStep.EMBED_ORIGINAL, "complete", f"Embedded {len(positives)+len(negatives)} texts → dim={original_emb.shape[1]}", "embedding",
                     shape=list(original_emb.shape))

                # Apply transforms (LLM)
                emit(EvalStep.APPLY_TRANSFORMS, "start", f"Applying transforms: {[t.name for t in transform_set.transforms if isinstance(t, Transform)]}", "llm")
                transformed_texts = []
                transforms_applied = []
                for text in positives:
                    t_text, t_names = self._apply_transforms(text, transform_set)
                    transformed_texts.append(t_text)
                    transforms_applied.extend(t_names)
                emit(EvalStep.APPLY_TRANSFORMS, "complete", f"Applied {len(set(transforms_applied))} unique transforms", "llm",
                     transforms=list(set(transforms_applied)), example=transformed_texts[0][:80] if transformed_texts else "")

                # Embed transformed (measurement)
                emit(EvalStep.EMBED_TRANSFORMED, "start", f"Embedding {len(transformed_texts)} transformed texts", "embedding")
                transformed_emb = self.embedding.embed(transformed_texts)
                emit(EvalStep.EMBED_TRANSFORMED, "complete", "Transformed embeddings ready", "embedding")

                results.append(TrialResult(
                    seed=seed,
                    original_embeddings=original_emb,
                    transformed_embeddings=transformed_emb,
                    negative_embeddings=negative_emb,
                    transforms_applied=list(set(transforms_applied)),
                ))
                emit(EvalStep.TRIAL_END, "complete", f"Trial {i+1} complete", "orchestrator")

            except Exception as e:
                emit(EvalStep.TRIAL_END, "error", f"Trial {i+1} failed: {e}", "orchestrator", error=str(e))
                if spec.on_trial_error.action == "abort":
                    raise
                results.append(TrialResult(
                    seed=seed,
                    original_embeddings=np.array([]),
                    transformed_embeddings=np.array([]),
                    negative_embeddings=np.array([]),
                    transforms_applied=[],
                    error=str(e),
                ))

        return [r for r in results if r.error is None]

    def _apply_transforms(
        self,
        text: str,
        transform_set: TransformSet,
    ) -> tuple[str, list[str]]:
        """Apply transforms using LLM provider."""
        transforms = [t for t in transform_set.transforms if isinstance(t, Transform)]

        if not transforms:
            return text, []

        # Select transforms
        policy = transform_set.policy
        n = min(policy.per_item, len(transforms))
        selected = random.sample(transforms, n)

        # Apply chain
        result = text
        applied = []
        for t in selected[:policy.compose_depth]:
            result = self.llm.apply_transform(result, t)
            applied.append(t.name)

        return result, applied

    def _resolve_transform_set(self, ts: TransformSet | ArtifactRef) -> TransformSet:
        """Resolve transform set reference."""
        if isinstance(ts, TransformSet):
            return ts
        # TODO: Load from store
        raise NotImplementedError("Loading transform sets from refs")

    # =========================================================================
    # DEFAULT SCOREBOARD
    # =========================================================================

    def _default_scoreboard(self, hypothesis: Hypothesis) -> ScoreboardSpec:
        """Create default scoreboard for a hypothesis."""
        return ScoreboardSpec(
            name="default_v1",
            lens=Lens(
                name="default_lens",
                model=ModelReference(
                    provider="stub",
                    model_id="stub-embedding-v1",
                    embedding_dim=128,
                ),
            ),
            trial_plan=TrialPlan(
                num_trials=3,
                seeds=[42, 123, 456],
            ),
            negatives=NegativeDataset(
                name="generated",
                source="generated",
                generator=NegativeGeneratorConfig(
                    strategy=NegativeStrategy.NEAR_MISS,
                    num_per_positive=2,
                ),
            ),
            transform_set=self._default_transform_set(),
            metrics=[
                MetricComputation(name=MetricName.STABILITY),
                MetricComputation(name=MetricName.MUTUAL_INFORMATION),
                MetricComputation(name=MetricName.KL_DRIFT),
                MetricComputation(name=MetricName.SEPARABILITY),
                MetricComputation(name=MetricName.REIDENTIFIABILITY),
            ],
            admissibility=AdmissibilityConfig(
                must_hold=[
                    AdmissibilityRule(
                        name="preserve_evaluation",
                        condition=ThresholdRule(
                            metric=MetricName.SEPARABILITY,
                            op=ComparisonOp.GTE,
                            value=0.55,
                        ),
                    ),
                    AdmissibilityRule(
                        name="keep_mi_positive",
                        condition=ThresholdRule(
                            metric=MetricName.MUTUAL_INFORMATION,
                            op=ComparisonOp.GTE,
                            value=0.05,
                        ),
                    ),
                    # Note: no_collapse removed - high separability is good
                    # (means negatives are distinguishable). Low sep is handled
                    # by falsification rules (NON_DISTINGUISHABILITY).
                ],
            ),
            decision=DecisionPolicy(
                falsify_if_any=[
                    FalsificationRule(
                        step=FailureMode.IDENTITY_INSTABILITY,
                        condition=ThresholdRule(
                            metric=MetricName.STABILITY,
                            op=ComparisonOp.LT,
                            value=0.65,
                        ),
                    ),
                    FalsificationRule(
                        step=FailureMode.NON_DISTINGUISHABILITY,
                        condition=ThresholdRule(
                            metric=MetricName.SEPARABILITY,
                            op=ComparisonOp.LT,
                            value=0.60,
                        ),
                    ),
                ],
                discover_if_all=[
                    ThresholdRule(
                        metric=MetricName.STABILITY,
                        op=ComparisonOp.GTE,
                        value=self.config.stability_threshold,
                    ),
                    ThresholdRule(
                        metric=MetricName.SEPARABILITY,
                        op=ComparisonOp.GTE,
                        value=self.config.separability_threshold,
                    ),
                    ThresholdRule(
                        metric=MetricName.KL_DRIFT,
                        op=ComparisonOp.LTE,
                        value=self.config.kl_max,
                    ),
                ],
            ),
        )

    def _default_transform_set(self) -> TransformSet:
        """Create default transform set.

        Semantic transforms that preserve meaning:
        - paraphrase: Different words, same meaning (core test)
        - formalize: Academic/formal register
        - simplify: Plain language version
        - add_context: Neutral surrounding context
        - abstract: More general statement
        """
        return TransformSet(
            name="semantic_transforms",
            transforms=[
                # Primary: paraphrase (core semantic test)
                Transform(
                    name="paraphrase",
                    category=TransformCategory.REPRESENTATION,
                    severity=0.5,
                    description="Rewrite with different words, same meaning",
                ),
                # Register shifts
                Transform(
                    name="formalize",
                    category=TransformCategory.REPRESENTATION,
                    severity=0.3,
                    description="Convert to formal/academic language",
                ),
                Transform(
                    name="simplify",
                    category=TransformCategory.REPRESENTATION,
                    severity=0.3,
                    description="Convert to simpler language",
                ),
                # Context transforms
                Transform(
                    name="add_context",
                    category=TransformCategory.CONTEXT,
                    severity=0.2,
                    description="Add neutral surrounding context",
                ),
                Transform(
                    name="abstract",
                    category=TransformCategory.CONTEXT,
                    severity=0.4,
                    description="Make more abstract/general",
                ),
            ],
        )

    # =========================================================================
    # RESULT PACKAGING
    # =========================================================================

    def _package_result(
        self,
        hypothesis: Hypothesis,
        spec: ScoreboardSpec,
        metrics: InvarianceScore,
        admissibility: AdmissibilityResult,
        outcome: EvaluationOutcome,
        failure_mode: FailureMode | None,
        trials: list[TrialResult],
    ) -> EvaluationResult:
        """Package final result based on outcome."""
        run_ref = ArtifactRef(kind="RunRecord", id=uuid4())
        hyp_ref = ArtifactRef(kind="Hypothesis", id=hypothesis.id)

        if outcome == EvaluationOutcome.FALSIFIED:
            return EvaluationResult(
                run=run_ref,
                hypothesis=hyp_ref,
                outcome=outcome,
                falsification=FalsificationReport(
                    run=run_ref,
                    hypothesis=hyp_ref,
                    failure_mode=failure_mode or FailureMode.NON_INVARIANCE,
                    counterexample=self._find_counterexample(trials),
                    metrics=metrics,
                    explanation=self._explain_failure(failure_mode, metrics),
                    suggested_refinements=self.stats.diagnose_failure(metrics.to_dict()),
                ),
                admissibility=admissibility,
            )

        elif outcome == EvaluationOutcome.DISCOVERY:
            return EvaluationResult(
                run=run_ref,
                hypothesis=hyp_ref,
                outcome=outcome,
                discovery=DiscoveryReport(
                    run=run_ref,
                    hypothesis=hyp_ref,
                    invariant_formal=self._formalize(hypothesis, spec),
                    invariant_plain=self._to_plain(hypothesis, metrics),
                    lens_used=spec.lens,
                    transform_set_used=spec.transform_set.name if isinstance(spec.transform_set, TransformSet) else "ref",
                    metrics=metrics,
                    boundary_conditions=self._find_boundaries(trials),
                    confidence=self._compute_confidence(metrics),
                ),
                admissibility=admissibility,
            )

        else:  # UNDERDETERMINED
            return EvaluationResult(
                run=run_ref,
                hypothesis=hyp_ref,
                outcome=outcome,
                underdetermined=UnderdeterminedReport(
                    run=run_ref,
                    hypothesis=hyp_ref,
                    reason="Metrics between falsification and discovery thresholds",
                    metrics=metrics,
                    conflicting_signals=self._find_conflicts(metrics),
                    suggested_experiments=self._suggest_experiments(metrics, spec),
                ),
                admissibility=admissibility,
            )

    def _empty_result(
        self,
        hypothesis: Hypothesis,
        reason: str,
    ) -> EvaluationResult:
        """Create result for failed execution."""
        run_ref = ArtifactRef(kind="RunRecord", id=uuid4())
        hyp_ref = ArtifactRef(kind="Hypothesis", id=hypothesis.id)

        return EvaluationResult(
            run=run_ref,
            hypothesis=hyp_ref,
            outcome=EvaluationOutcome.UNDERDETERMINED,
            underdetermined=UnderdeterminedReport(
                run=run_ref,
                hypothesis=hyp_ref,
                reason=reason,
                metrics=InvarianceScore(metrics=[]),
                suggested_experiments=["Fix trial execution errors"],
            ),
        )

    def _inadmissible_result(
        self,
        hypothesis: Hypothesis,
        admissibility: AdmissibilityResult,
    ) -> EvaluationResult:
        """Create result for inadmissible transforms."""
        run_ref = ArtifactRef(kind="RunRecord", id=uuid4())
        hyp_ref = ArtifactRef(kind="Hypothesis", id=hypothesis.id)

        return EvaluationResult(
            run=run_ref,
            hypothesis=hyp_ref,
            outcome=EvaluationOutcome.INADMISSIBLE,
            admissibility=admissibility,
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _find_counterexample(self, trials: list[TrialResult]) -> MinimalCounterexample:
        """Find minimal counterexample from trials."""
        if not trials:
            return MinimalCounterexample(
                input_text="[no trials]",
                breaking_transform="none",
                transform_depth=0,
                original_output=None,
                transformed_output=None,
                divergence_metric=1.0,
            )

        worst = min(
            trials,
            key=lambda t: self.stats.stability(
                t.original_embeddings, t.transformed_embeddings
            ),
        )

        return MinimalCounterexample(
            input_text="[input extraction not yet implemented]",
            breaking_transform=worst.transforms_applied[0] if worst.transforms_applied else "none",
            transform_depth=len(worst.transforms_applied),
            original_output=None,
            transformed_output=None,
            divergence_metric=1.0 - self.stats.stability(
                worst.original_embeddings, worst.transformed_embeddings
            ),
        )

    def _find_boundaries(self, trials: list[TrialResult]) -> list[BoundaryCondition]:
        """Find boundary conditions."""
        boundaries = []
        for t in trials:
            stab = self.stats.stability(t.original_embeddings, t.transformed_embeddings)
            if stab < 0.9:
                boundaries.append(BoundaryCondition(
                    description=f"Stability drops to {stab:.2f}",
                    failing_transform=", ".join(t.transforms_applied),
                    threshold=stab,
                ))
        return boundaries[:5]

    def _find_conflicts(self, metrics: InvarianceScore) -> list[str]:
        """Find conflicting signals."""
        return self.stats.diagnose_failure(metrics.to_dict())

    def _suggest_experiments(
        self,
        metrics: InvarianceScore,
        spec: ScoreboardSpec,
    ) -> list[str]:
        """Suggest next experiments."""
        suggestions = []
        stab = metrics.get(MetricName.STABILITY) or 0
        if 0.6 < stab < 0.8:
            suggestions.append("Run more trials to reduce variance")
        suggestions.append("Test on additional model family")
        suggestions.append("Try milder or stronger transforms")
        return suggestions

    def _explain_failure(
        self,
        mode: FailureMode | None,
        metrics: InvarianceScore,
    ) -> str:
        """Generate failure explanation."""
        if mode == FailureMode.NON_INVARIANCE:
            stab = metrics.get(MetricName.STABILITY) or 0
            return f"Stability ({stab:.2f}) below threshold"
        elif mode == FailureMode.NON_DISTINGUISHABILITY:
            sep = metrics.get(MetricName.SEPARABILITY) or 0
            return f"Separability ({sep:.2f}) too low"
        return "Failed evaluation criteria"

    def _formalize(self, hypothesis: Hypothesis, spec: ScoreboardSpec) -> str:
        """Generate formal invariant statement."""
        ts_name = spec.transform_set.name if isinstance(spec.transform_set, TransformSet) else "ref"
        return (
            f"INVARIANT: {hypothesis.title}\n"
            f"UNDER: {ts_name}\n"
            f"LENS: {spec.lens.name}"
        )

    def _to_plain(self, hypothesis: Hypothesis, metrics: InvarianceScore) -> str:
        """Generate plain language description."""
        stab = metrics.get(MetricName.STABILITY) or 0
        sep = metrics.get(MetricName.SEPARABILITY) or 0
        return (
            f"'{hypothesis.title}' is stable ({stab:.0%}) under transforms "
            f"and distinguishable ({sep:.0%}) from alternatives."
        )

    def _compute_confidence(self, metrics: InvarianceScore) -> float:
        """Compute overall confidence."""
        values = [m.value for m in metrics.metrics if m.name != MetricName.KL_DRIFT]
        return float(np.mean(values)) if values else 0.0
