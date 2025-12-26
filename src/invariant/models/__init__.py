"""Pydantic schemas for core artifacts."""

from invariant.models.schemas import (
    # Enums
    HypothesisType,
    FailureMode,
    EvaluationOutcome,
    MetricName,
    ComparisonOp,
    NegativeStrategy,
    TransformCategory,
    # Provenance & References
    Provenance,
    ArtifactRef,
    # Model & Lens
    ModelReference,
    Lens,
    # Negatives
    NegativeGeneratorConfig,
    NegativeDataset,
    # Transforms
    TransformRisk,
    Transform,
    TransformApplicationPolicy,
    TransformSet,
    # Metrics
    MetricComputation,
    MetricValue,
    # Decision Rules
    ThresholdRule,
    FalsificationRule,
    AdmissibilityRule,
    DecisionPolicy,
    # Admissibility
    AdmissibilityConfig,
    AdmissibilityResult,
    # Scoreboard
    TrialPlan,
    CounterexamplePolicy,
    ReportingConfig,
    CheckpointPolicy,
    OnTrialErrorPolicy,
    ScoreboardSpec,
    # Hypothesis
    ClaimTarget,
    Hypothesis,
    # Experiment & Run
    ExperimentPlan,
    RunRecord,
    # Reports
    InvarianceScore,
    MinimalCounterexample,
    BoundaryCondition,
    FalsificationReport,
    DiscoveryReport,
    UnderdeterminedReport,
    EvaluationResult,
)

__all__ = [
    # Enums
    "HypothesisType",
    "FailureMode",
    "EvaluationOutcome",
    "MetricName",
    "ComparisonOp",
    "NegativeStrategy",
    "TransformCategory",
    # Provenance & References
    "Provenance",
    "ArtifactRef",
    # Model & Lens
    "ModelReference",
    "Lens",
    # Negatives
    "NegativeGeneratorConfig",
    "NegativeDataset",
    # Transforms
    "TransformRisk",
    "Transform",
    "TransformApplicationPolicy",
    "TransformSet",
    # Metrics
    "MetricComputation",
    "MetricValue",
    # Decision Rules
    "ThresholdRule",
    "FalsificationRule",
    "AdmissibilityRule",
    "DecisionPolicy",
    # Admissibility
    "AdmissibilityConfig",
    "AdmissibilityResult",
    # Scoreboard
    "TrialPlan",
    "CounterexamplePolicy",
    "ReportingConfig",
    "CheckpointPolicy",
    "OnTrialErrorPolicy",
    "ScoreboardSpec",
    # Hypothesis
    "ClaimTarget",
    "Hypothesis",
    # Experiment & Run
    "ExperimentPlan",
    "RunRecord",
    # Reports
    "InvarianceScore",
    "MinimalCounterexample",
    "BoundaryCondition",
    "FalsificationReport",
    "DiscoveryReport",
    "UnderdeterminedReport",
    "EvaluationResult",
]
