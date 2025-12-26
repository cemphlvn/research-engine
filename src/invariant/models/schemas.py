"""Core data models for Invariant engine.

Incorporates:
- Pydantic v2 for runtime validation
- ScoreboardSpec as executable evaluation DSL
- Explicit lens/representation versioning
- Mandatory negative datasets
- Structured decision logic (no string eval)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


def _utc_now() -> datetime:
    """UTC now with timezone info (replaces deprecated utcnow)."""
    return datetime.now(timezone.utc)


# =============================================================================
# ENUMS (closed sets)
# =============================================================================


class HypothesisType(str, Enum):
    """What kind of structure is being claimed."""

    CONCEPT_INVARIANT = "concept_invariant"  # A concept has stable structure
    RELATION_LAW = "relation_law"  # R(x,y) persists under transforms
    CLUSTER_STRUCTURE = "cluster_structure"  # Grouping is stable
    CAUSAL_RULE = "causal_rule"  # X causes Y
    SYMMETRY = "symmetry"  # Invariance under specific group


class FailureMode(str, Enum):
    """Classification of how a hypothesis failed."""

    SELF_CONTRADICTION = "self_contradiction"
    IDENTITY_INSTABILITY = "identity_instability"
    NON_DISTINGUISHABILITY = "non_distinguishability"
    LACK_OF_CONSTRAINT = "lack_of_constraint"
    NON_REIDENTIFIABILITY = "non_reidentifiability"
    NON_INVARIANCE = "non_invariance"
    INADMISSIBLE_TRANSFORMS = "inadmissible_transforms"


class EvaluationOutcome(str, Enum):
    """Final verdict."""

    DISCOVERY = "discovery"
    FALSIFIED = "falsified"
    UNDERDETERMINED = "underdetermined"
    INADMISSIBLE = "inadmissible"  # Added: transforms failed admissibility


class MetricName(str, Enum):
    """Known metric types."""

    STABILITY = "stability"
    MUTUAL_INFORMATION = "mutual_information"
    KL_DRIFT = "kl_drift"
    SEPARABILITY = "separability"
    REIDENTIFIABILITY = "reidentifiability"
    COMPRESSION_GAIN = "compression_gain"  # Deferred but defined


class ComparisonOp(str, Enum):
    """Comparison operators for decision rules."""

    GTE = ">="
    GT = ">"
    LTE = "<="
    LT = "<"
    EQ = "=="
    NEQ = "!="


class NegativeStrategy(str, Enum):
    """How to generate negative examples.

    Semantic strategies (use these for real evaluation):
    - NEAR_MISS: Same topic, subtly different meaning (hardest)
    - ANTONYM: Replace key concepts with opposites
    - CONTRADICT: Logically contradictory statements
    - RELATED: Related topic but different claim
    - BOUNDARY: Edge cases that almost satisfy but don't

    Non-semantic (avoid for real evaluation):
    - RANDOM: Completely unrelated (too easy)
    - SHUFFLE: Word permutation (syntactic only)
    """

    # Semantic strategies (preferred)
    NEAR_MISS = "near_miss"  # Same topic, subtle semantic difference
    ANTONYM = "antonym"  # Key concepts replaced with opposites
    CONTRADICT = "contradict"  # Logically contradictory
    RELATED = "related"  # Related topic, different claim
    BOUNDARY = "boundary"  # Edge cases, boundary violations

    # Legacy/testing (avoid for real evaluation)
    RANDOM = "random"  # Unrelated sentences (too easy to distinguish)
    SHUFFLE = "shuffle"  # Word permutation (syntactic only)
    NEGATE = "negate"  # Simple negation prefix
    MANUAL = "manual"  # User-provided


class TransformCategory(str, Enum):
    """Transform families (not rigid types)."""

    REPRESENTATION = "representation"  # paraphrase, synonym, encoding
    CONTEXT = "context"  # add/remove context, distractors
    SAMPLING = "sampling"  # temperature, seed variation
    STRUCTURAL = "structural"  # graph ops, permutations


# =============================================================================
# PROVENANCE & REFERENCES
# =============================================================================


class Provenance(BaseModel):
    """Audit trail for any artifact."""

    created_at: datetime = Field(default_factory=_utc_now)
    created_by: str = "engine"
    source: str | None = None
    tags: list[str] = Field(default_factory=list)


class ArtifactRef(BaseModel):
    """Reference to another artifact by kind + id."""

    kind: str
    id: UUID


# =============================================================================
# MODEL & LENS (representation versioning)
# =============================================================================


class ModelReference(BaseModel):
    """Pinned reference to a model for reproducibility."""

    provider: str  # "anthropic", "openai", "local"
    model_id: str  # "claude-3-sonnet", "text-embedding-3-large"
    version: str | None = None  # API version or checkpoint
    embedding_dim: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)  # temperature, etc.


class Lens(BaseModel):
    """The representation through which invariance is measured.

    Invariance is always lens-relative. Different lens = different invariants.
    """

    name: str
    model: ModelReference
    normalize: bool = True
    pooling: str = "mean"  # mean, cls, last
    version: str = "v1"  # Lens config version


# =============================================================================
# NEGATIVE DATASETS (falsifiability requires teeth)
# =============================================================================


class NegativeGeneratorConfig(BaseModel):
    """How to generate negative examples."""

    strategy: NegativeStrategy
    num_per_positive: int = 2
    params: dict[str, Any] = Field(default_factory=dict)


class NegativeDataset(BaseModel):
    """First-class artifact for negative examples.

    Without negatives, everything is 'true'.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    source: Literal["generated", "manual", "hybrid"]
    generator: NegativeGeneratorConfig | None = None
    items: list[str] = Field(default_factory=list)  # If manual/hybrid
    provenance: Provenance = Field(default_factory=Provenance)


# =============================================================================
# TRANSFORMS
# =============================================================================


class TransformRisk(BaseModel):
    """Risk associated with a transform."""

    severity: Literal["warning", "error", "critical"]
    message: str


class Transform(BaseModel):
    """A single transformation with metadata."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    category: TransformCategory
    family: str | None = None  # Flexible sub-grouping
    params: dict[str, Any] = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    expected_invariants: list[str] = Field(default_factory=list)
    risks: list[TransformRisk] = Field(default_factory=list)
    severity: float = Field(ge=0.0, le=1.0, default=0.5)
    version: str = "v1"  # Transform version
    provenance: Provenance = Field(default_factory=Provenance)


class TransformApplicationPolicy(BaseModel):
    """How to apply transforms."""

    per_item: int = 4  # Transforms per input
    compose_depth: int = 2  # Max chain length
    random_order: bool = True


class TransformSet(BaseModel):
    """A collection of transforms with application policy."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    transforms: list[Transform | ArtifactRef]
    policy: TransformApplicationPolicy = Field(default_factory=TransformApplicationPolicy)
    description: str = ""
    provenance: Provenance = Field(default_factory=Provenance)


# =============================================================================
# METRICS (structured, not string DSL)
# =============================================================================


class MetricComputation(BaseModel):
    """Structured metric computation (no string eval)."""

    name: MetricName
    inputs: list[str] = Field(default_factory=lambda: ["f_original", "f_transformed"])
    aggregate: Literal["mean", "median", "min", "max", "std"] = "mean"
    params: dict[str, Any] = Field(default_factory=dict)


class MetricValue(BaseModel):
    """A computed metric value with uncertainty."""

    name: MetricName
    value: float
    ci95: tuple[float, float] | None = None
    n_samples: int | None = None
    notes: str | None = None


# =============================================================================
# DECISION RULES (structured, not string expressions)
# =============================================================================


class ThresholdRule(BaseModel):
    """A single threshold condition."""

    metric: MetricName
    op: ComparisonOp
    value: float

    def evaluate(self, metrics: dict[MetricName, float]) -> bool:
        """Evaluate this rule against computed metrics."""
        if self.metric not in metrics:
            return False
        actual = metrics[self.metric]
        ops = {
            ComparisonOp.GTE: lambda a, b: a >= b,
            ComparisonOp.GT: lambda a, b: a > b,
            ComparisonOp.LTE: lambda a, b: a <= b,
            ComparisonOp.LT: lambda a, b: a < b,
            ComparisonOp.EQ: lambda a, b: a == b,
            ComparisonOp.NEQ: lambda a, b: a != b,
        }
        return ops[self.op](actual, self.value)


class FalsificationRule(BaseModel):
    """A rule that triggers falsification."""

    step: FailureMode
    condition: ThresholdRule


class AdmissibilityRule(BaseModel):
    """A rule that must hold for transforms to be admissible."""

    name: str  # "preserve_evaluation", "preserve_falsifiability", etc.
    condition: ThresholdRule
    required: bool = True


class DecisionPolicy(BaseModel):
    """The decision logic: when to falsify, discover, or remain underdetermined."""

    falsify_if_any: list[FalsificationRule] = Field(default_factory=list)
    discover_if_all: list[ThresholdRule] = Field(default_factory=list)
    otherwise: Literal["underdetermined"] = "underdetermined"


# =============================================================================
# ADMISSIBILITY
# =============================================================================


class AdmissibilityConfig(BaseModel):
    """Configuration for admissibility checking."""

    must_hold: list[AdmissibilityRule]
    on_fail_outcome: Literal["inadmissible"] = "inadmissible"
    mi_threshold: float = 0.05  # Minimum mutual information


class AdmissibilityResult(BaseModel):
    """Result of admissibility check."""

    admissible: bool
    rule_results: dict[str, bool] = Field(default_factory=dict)
    failing_rules: list[str] = Field(default_factory=list)
    mi_actual: float | None = None
    suggestions: list[str] = Field(default_factory=list)


# =============================================================================
# SCOREBOARD SPEC (executable evaluation DSL)
# =============================================================================


class TrialPlan(BaseModel):
    """Reproducible trial configuration."""

    num_trials: int = 64
    seeds: list[int]  # Explicit, not "auto"
    models: list[str] = Field(default_factory=lambda: ["default"])
    sampling_params: dict[str, Any] = Field(default_factory=dict)  # Generalized from temperatures

    @field_validator("seeds")
    @classmethod
    def seeds_not_empty(cls, v):
        if not v:
            raise ValueError("seeds must be explicitly provided (no 'auto')")
        return v


class CounterexamplePolicy(BaseModel):
    """How to find minimal counterexamples."""

    enabled: bool = True
    minimize: list[str] = Field(default_factory=lambda: ["input_length", "transform_depth"])
    max_search_iterations: int = 100


class ReportingConfig(BaseModel):
    """What to include in reports."""

    emit_counterexample: bool = True
    counterexample_policy: CounterexamplePolicy = Field(default_factory=CounterexamplePolicy)
    include_boundaries: bool = True
    include_per_trial_metrics: bool = False


class CheckpointPolicy(BaseModel):
    """When to save intermediate state."""

    every_n_trials: int = 10
    on_failure: bool = True
    storage_path: str = ".invariant/checkpoints"


class OnTrialErrorPolicy(BaseModel):
    """What to do when a trial fails."""

    action: Literal["skip", "retry", "abort"] = "skip"
    max_retries: int = 3
    log_errors: bool = True


class ScoreboardSpec(BaseModel):
    """The executable evaluation specification.

    This is the 'scoreboard' - the public, explicit rules of winning.
    If you can't write the scoreboard, you can't claim knowledge.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str
    version: str = "v1.0.0"

    # What lens to use
    lens: Lens

    # How to run trials
    trial_plan: TrialPlan

    # Negative examples (mandatory for falsifiability)
    negatives: NegativeDataset | ArtifactRef

    # Transforms to apply
    transform_set: TransformSet | ArtifactRef

    # What to measure
    metrics: list[MetricComputation]

    # Admissibility gates
    admissibility: AdmissibilityConfig

    # Decision logic
    decision: DecisionPolicy

    # Reporting configuration
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    # Error handling
    on_trial_error: OnTrialErrorPolicy = Field(default_factory=OnTrialErrorPolicy)

    # Checkpointing
    checkpoint: CheckpointPolicy = Field(default_factory=CheckpointPolicy)

    provenance: Provenance = Field(default_factory=Provenance)


# =============================================================================
# HYPOTHESIS
# =============================================================================


class ClaimTarget(BaseModel):
    """What the hypothesis is about."""

    mode: Literal["text", "dataset_ref", "embedding_ref", "model_behavior_ref"]
    content: str | dict[str, Any]


class Hypothesis(BaseModel):
    """A candidate invariant structure to test."""

    id: UUID = Field(default_factory=uuid4)
    title: str
    target: ClaimTarget
    hypothesis_type: HypothesisType
    formalization: str | dict[str, Any] | None = None
    predicted_invariants: list[str] = Field(default_factory=list)
    expected_failure_modes: list[FailureMode] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)


# =============================================================================
# EXPERIMENT & RUN
# =============================================================================


class ExperimentPlan(BaseModel):
    """Links hypothesis + scoreboard for execution."""

    id: UUID = Field(default_factory=uuid4)
    hypothesis: Hypothesis | ArtifactRef
    scoreboard: ScoreboardSpec | ArtifactRef
    provenance: Provenance = Field(default_factory=Provenance)


class RunRecord(BaseModel):
    """For reproducibility."""

    id: UUID = Field(default_factory=uuid4)
    plan: ExperimentPlan | ArtifactRef
    status: Literal["running", "success", "failed", "canceled"]
    started_at: datetime = Field(default_factory=_utc_now)
    finished_at: datetime | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)
    checkpoints: list[str] = Field(default_factory=list)


# =============================================================================
# REPORTS
# =============================================================================


class InvarianceScore(BaseModel):
    """Quantified invariance metrics."""

    metrics: list[MetricValue]

    def get(self, name: MetricName) -> float | None:
        for m in self.metrics:
            if m.name == name:
                return m.value
        return None

    def to_dict(self) -> dict[MetricName, float]:
        return {m.name: m.value for m in self.metrics}


class MinimalCounterexample(BaseModel):
    """Smallest input/transform combo that breaks hypothesis."""

    input_text: str
    breaking_transform: str
    transform_depth: int
    original_output: Any
    transformed_output: Any
    divergence_metric: float


class BoundaryCondition(BaseModel):
    """Where an invariant stops working."""

    description: str
    failing_transform: str | None = None
    threshold: float | None = None


class FalsificationReport(BaseModel):
    """Detailed failure analysis."""

    id: UUID = Field(default_factory=uuid4)
    run: ArtifactRef
    hypothesis: ArtifactRef
    failure_mode: FailureMode
    failed_rule: FalsificationRule | None = None
    counterexample: MinimalCounterexample | None = None
    metrics: InvarianceScore
    explanation: str
    suggested_refinements: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)


class DiscoveryReport(BaseModel):
    """Packaged invariant that survived testing."""

    id: UUID = Field(default_factory=uuid4)
    run: ArtifactRef
    hypothesis: ArtifactRef
    invariant_formal: str
    invariant_plain: str
    lens_used: Lens
    transform_set_used: str
    metrics: InvarianceScore
    boundary_conditions: list[BoundaryCondition] = Field(default_factory=list)
    minimal_basis: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: Provenance = Field(default_factory=Provenance)


class UnderdeterminedReport(BaseModel):
    """Insufficient evidence to decide."""

    id: UUID = Field(default_factory=uuid4)
    run: ArtifactRef
    hypothesis: ArtifactRef
    reason: str  # Free text, not rigid enum
    metrics: InvarianceScore
    conflicting_signals: list[str] = Field(default_factory=list)
    suggested_experiments: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)


class EvaluationResult(BaseModel):
    """Final output of the pipeline."""

    id: UUID = Field(default_factory=uuid4)
    run: ArtifactRef
    hypothesis: ArtifactRef
    outcome: EvaluationOutcome
    discovery: DiscoveryReport | None = None
    falsification: FalsificationReport | None = None
    underdetermined: UnderdeterminedReport | None = None
    admissibility: AdmissibilityResult | None = None
