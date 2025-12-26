"""Stats engine: The epistemic court.

This layer is the hard gatekeeper. LLMs generate, stats decide.
If you remove this layer, you get a storytelling engine.
If you keep it, you get a falsifiable discovery engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from invariant.models.schemas import (
    AdmissibilityConfig,
    AdmissibilityResult,
    DecisionPolicy,
    EvaluationOutcome,
    FailureMode,
    InvarianceScore,
    MetricComputation,
    MetricName,
    MetricValue,
    ThresholdRule,
)
from invariant.utils.info_theory import (
    entropy,
    kl_divergence,
    mutual_information,
    stability_score,
    info_preserved_ratio,
)


@dataclass
class StatsConfig:
    """Configuration for stats engine."""

    n_bootstrap: int = 100
    confidence_level: float = 0.95
    min_samples_for_ci: int = 10


class StatsEngine:
    """The epistemic court - statistical validation of invariance claims.

    Responsibilities:
    1. Compute invariance metrics (stability, MI, KL, separability)
    2. Quantify uncertainty (bootstrap CIs)
    3. Check admissibility gates
    4. Apply decision logic
    5. Detect scoreboard collapse

    This is what makes it "research" instead of "prompt engineering".
    """

    def __init__(self, config: StatsConfig | None = None):
        self.config = config or StatsConfig()

    # =========================================================================
    # METRIC COMPUTATION
    # =========================================================================

    def compute_metrics(
        self,
        original: NDArray[np.float64],
        transformed: NDArray[np.float64],
        negatives: NDArray[np.float64],
        metric_specs: list[MetricComputation],
    ) -> InvarianceScore:
        """Compute all requested metrics with uncertainty estimates.

        Args:
            original: Original embeddings (N, D)
            transformed: Transformed embeddings (N, D)
            negatives: Negative example embeddings (M, D)
            metric_specs: Which metrics to compute

        Returns:
            InvarianceScore with values and CIs
        """
        metrics = []

        for spec in metric_specs:
            value = self._compute_single_metric(
                spec.name, original, transformed, negatives
            )
            ci = self._bootstrap_ci(
                spec.name, original, transformed, negatives
            )

            metrics.append(MetricValue(
                name=spec.name,
                value=value,
                ci95=ci,
                n_samples=len(original),
            ))

        return InvarianceScore(metrics=metrics)

    def _compute_single_metric(
        self,
        name: MetricName,
        original: NDArray[np.float64],
        transformed: NDArray[np.float64],
        negatives: NDArray[np.float64],
    ) -> float:
        """Compute a single metric value."""
        if name == MetricName.STABILITY:
            return self.stability(original, transformed)

        elif name == MetricName.MUTUAL_INFORMATION:
            return self.mutual_info_preserved(original, transformed)

        elif name == MetricName.KL_DRIFT:
            return self.kl_drift(original, transformed)

        elif name == MetricName.SEPARABILITY:
            return self.separability(transformed, negatives)

        elif name == MetricName.REIDENTIFIABILITY:
            return self.reidentifiability(original, transformed)

        elif name == MetricName.COMPRESSION_GAIN:
            return 0.0  # Not yet implemented

        return 0.0

    # =========================================================================
    # CORE METRICS
    # =========================================================================

    def stability(
        self,
        original: NDArray[np.float64],
        transformed: NDArray[np.float64],
    ) -> float:
        """Measure stability: do representations stay similar under transform?

        High stability = invariant structure exists.
        """
        return stability_score(original, transformed)

    def mutual_info_preserved(
        self,
        original: NDArray[np.float64],
        transformed: NDArray[np.float64],
    ) -> float:
        """Measure information preservation: I(f(X); f(T(X))) / H(f(X)).

        High MI = structure survives transformation.
        """
        return info_preserved_ratio(original, transformed)

    def kl_drift(
        self,
        original: NDArray[np.float64],
        transformed: NDArray[np.float64],
    ) -> float:
        """Measure distribution drift: D_KL(P(original) || P(transformed)).

        High drift = your invariance is an illusion.
        Uses PCA to find axis of maximum variance for projection.
        """
        if len(original) < 3 or len(transformed) < 3:
            return 0.0

        # Combine both sets to find axis of maximum variance
        combined = np.vstack([original, transformed])
        combined_centered = combined - combined.mean(axis=0)

        # Use SVD to get first principal component
        try:
            _, _, Vt = np.linalg.svd(combined_centered, full_matrices=False)
            axis = Vt[0]  # First principal component
        except np.linalg.LinAlgError:
            return 0.0

        # Project onto principal axis
        orig_proj = original @ axis
        trans_proj = transformed @ axis

        # Check for constant projections
        if np.std(orig_proj) < 1e-10 and np.std(trans_proj) < 1e-10:
            return 0.0  # Both constant = no drift measurable

        # Use combined range for bins to handle distribution shift
        all_proj = np.concatenate([orig_proj, trans_proj])
        n_bins = min(30, max(3, len(orig_proj) // 2))
        bins = np.linspace(all_proj.min() - 1e-10, all_proj.max() + 1e-10, n_bins + 1)

        hist_orig, _ = np.histogram(orig_proj, bins=bins)
        hist_trans, _ = np.histogram(trans_proj, bins=bins)

        # Normalize manually (safer than density=True)
        hist_orig = hist_orig.astype(np.float64)
        hist_trans = hist_trans.astype(np.float64)

        if hist_orig.sum() < 1e-10 or hist_trans.sum() < 1e-10:
            return 0.0

        hist_orig = hist_orig / hist_orig.sum()
        hist_trans = hist_trans / hist_trans.sum()

        return kl_divergence(hist_orig, hist_trans)

    def separability(
        self,
        positives: NDArray[np.float64],
        negatives: NDArray[np.float64],
    ) -> float:
        """Measure separability: can we distinguish positives from negatives?

        AUC of simple classifier. Without this, everything is "true".
        """
        if len(positives) < 3 or len(negatives) < 3:
            return 1.0  # Not enough data

        X = np.vstack([positives, negatives])
        y = np.array([1] * len(positives) + [0] * len(negatives))

        try:
            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            cv_folds = min(5, min(len(positives), len(negatives)))
            if cv_folds < 2:
                return 1.0
            scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="roc_auc")
            return float(np.mean(scores))
        except Exception:
            return 0.5

    def reidentifiability(
        self,
        original: NDArray[np.float64],
        transformed: NDArray[np.float64],
    ) -> float:
        """Measure reidentifiability: same top patterns across runs?

        High = robust finding. Low = random artifact.
        """
        # Compute pairwise similarities
        orig_sim = original @ original.T
        trans_sim = transformed @ transformed.T

        # Check if ranking is preserved
        orig_ranks = np.argsort(orig_sim, axis=1)
        trans_ranks = np.argsort(trans_sim, axis=1)

        # Kendall-like correlation
        agreements = 0
        total = 0
        for i in range(len(orig_ranks)):
            for j in range(len(orig_ranks[i]) - 1):
                for k in range(j + 1, len(orig_ranks[i])):
                    orig_order = orig_ranks[i, j] < orig_ranks[i, k]
                    trans_order = trans_ranks[i, j] < trans_ranks[i, k]
                    if orig_order == trans_order:
                        agreements += 1
                    total += 1

        return agreements / total if total > 0 else 1.0

    # =========================================================================
    # UNCERTAINTY QUANTIFICATION
    # =========================================================================

    def _bootstrap_ci(
        self,
        metric: MetricName,
        original: NDArray[np.float64],
        transformed: NDArray[np.float64],
        negatives: NDArray[np.float64],
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for a metric."""
        if len(original) < self.config.min_samples_for_ci:
            return (0.0, 1.0)

        values = []
        n = len(original)

        for _ in range(self.config.n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            orig_sample = original[idx]
            trans_sample = transformed[idx]

            # Sample negatives too
            neg_idx = np.random.choice(len(negatives), size=len(negatives), replace=True)
            neg_sample = negatives[neg_idx]

            value = self._compute_single_metric(
                metric, orig_sample, trans_sample, neg_sample
            )
            values.append(value)

        lower = float(np.percentile(values, 100 * alpha / 2))
        upper = float(np.percentile(values, 100 * (1 - alpha / 2)))

        return (lower, upper)

    # =========================================================================
    # ADMISSIBILITY CHECKING
    # =========================================================================

    def check_admissibility(
        self,
        config: AdmissibilityConfig,
        metrics: dict[MetricName, float],
    ) -> AdmissibilityResult:
        """Check all admissibility rules.

        Admissibility = transforms preserve evaluation + falsifiability.
        """
        rule_results = {}
        failing_rules = []
        suggestions = []

        for rule in config.must_hold:
            passed = rule.condition.evaluate(metrics)
            rule_results[rule.name] = passed

            if not passed and rule.required:
                failing_rules.append(rule.name)
                suggestions.append(
                    f"Rule '{rule.name}' failed: "
                    f"{rule.condition.metric.value} {rule.condition.op.value} {rule.condition.value}"
                )

        # Note: no_scoreboard_collapse check removed - high separability
        # (distinguishable negatives) is good. Use detect_collapse() for diagnostics.

        return AdmissibilityResult(
            admissible=len(failing_rules) == 0,
            rule_results=rule_results,
            failing_rules=failing_rules,
            mi_actual=metrics.get(MetricName.MUTUAL_INFORMATION),
            suggestions=suggestions,
        )

    # =========================================================================
    # DECISION LOGIC
    # =========================================================================

    def apply_decision(
        self,
        policy: DecisionPolicy,
        metrics: dict[MetricName, float],
    ) -> tuple[EvaluationOutcome, FailureMode | None]:
        """Apply decision logic to determine outcome.

        Order:
        1. Check falsify_if_any - any trigger = FALSIFIED
        2. Check discover_if_all - all pass = DISCOVERY
        3. Otherwise = UNDERDETERMINED
        """
        # Check falsification rules first
        for rule in policy.falsify_if_any:
            if rule.condition.evaluate(metrics):
                return EvaluationOutcome.FALSIFIED, rule.step

        # Check discovery rules
        all_pass = all(
            rule.evaluate(metrics)
            for rule in policy.discover_if_all
        )

        if all_pass and policy.discover_if_all:
            return EvaluationOutcome.DISCOVERY, None

        return EvaluationOutcome.UNDERDETERMINED, None

    # =========================================================================
    # DIAGNOSTIC UTILITIES
    # =========================================================================

    def detect_collapse(
        self,
        metrics: dict[MetricName, float],
    ) -> tuple[bool, str]:
        """Detect if scoreboard has collapsed (everything passes).

        Collapse = your engine will "discover" whatever you want.
        """
        sep = metrics.get(MetricName.SEPARABILITY, 0)
        mi = metrics.get(MetricName.MUTUAL_INFORMATION, 0)
        stab = metrics.get(MetricName.STABILITY, 0)

        if sep > 0.98 and stab > 0.98:
            return True, "All metrics near 1.0 - likely collapsed or trivial"

        if mi < 0.01:
            return True, "MI near 0 - transforms may be destroying all information"

        return False, ""

    def diagnose_failure(
        self,
        metrics: dict[MetricName, float],
    ) -> list[str]:
        """Diagnose what's wrong when things fail."""
        issues = []

        stab = metrics.get(MetricName.STABILITY, 0)
        sep = metrics.get(MetricName.SEPARABILITY, 0)
        mi = metrics.get(MetricName.MUTUAL_INFORMATION, 0)
        kl = metrics.get(MetricName.KL_DRIFT, 0)

        if stab < 0.5:
            issues.append(f"Very low stability ({stab:.2f}) - claim may be too broad")

        if sep < 0.6:
            issues.append(f"Low separability ({sep:.2f}) - can't distinguish from negatives")

        if mi < 0.3:
            issues.append(f"Low MI preserved ({mi:.2f}) - transforms destroying structure")

        if kl > 0.5:
            issues.append(f"High KL drift ({kl:.2f}) - distribution shift under transform")

        if stab > 0.9 and sep < 0.6:
            issues.append("High stability but low separability - maybe vacuously stable")

        return issues
