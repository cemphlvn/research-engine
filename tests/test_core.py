"""Core tests for invariant engine."""

import numpy as np
import pytest

from invariant.core.orchestrator import Orchestrator, OrchestratorConfig, EvalStep
from invariant.models.schemas import (
    HypothesisType,
    EvaluationOutcome,
    MetricName,
    NegativeStrategy,
)
from invariant.providers.base import StubLLMProvider, StubEmbeddingProvider, ProviderConfig
from invariant.providers.stats import StatsEngine, StatsConfig
from invariant.utils.info_theory import entropy, mutual_information, stability_score, kl_divergence


class TestInfoTheory:
    """Test information-theoretic primitives."""

    def test_entropy_uniform(self):
        """Uniform distribution has max entropy."""
        x = np.random.uniform(0, 1, 1000)
        h = entropy(x)
        assert h > 0, "Entropy should be positive for non-constant"

    def test_entropy_constant(self):
        """Constant distribution has zero entropy."""
        x = np.ones(100)
        h = entropy(x)
        assert h == 0.0, "Constant should have 0 entropy"

    def test_entropy_insufficient_data(self):
        """Single point returns 0."""
        x = np.array([1.0])
        h = entropy(x)
        assert h == 0.0

    def test_stability_identical(self):
        """Identical vectors have stability 1.0."""
        x = np.random.randn(10, 128)
        s = stability_score(x, x)
        assert s == pytest.approx(1.0, abs=0.01)

    def test_stability_opposite(self):
        """Opposite vectors have stability ~0."""
        x = np.random.randn(10, 128)
        s = stability_score(x, -x)
        assert s < 0.1

    def test_kl_divergence_identical(self):
        """KL of identical dists is near 0."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        kl = kl_divergence(p, p)
        assert kl == pytest.approx(0.0, abs=0.01)

    def test_kl_divergence_different(self):
        """KL of different dists is positive."""
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        kl = kl_divergence(p, q)
        assert kl > 0.5


class TestStatsEngine:
    """Test the epistemic court."""

    @pytest.fixture
    def stats(self):
        return StatsEngine(StatsConfig(n_bootstrap=10))

    def test_stability_computation(self, stats):
        """Stability metric works."""
        orig = np.random.randn(10, 64)
        trans = orig + np.random.randn(10, 64) * 0.1  # Small perturbation
        s = stats.stability(orig, trans)
        assert 0 <= s <= 1

    def test_separability_distinguishable(self, stats):
        """Separable clusters have high AUC."""
        pos = np.random.randn(50, 32) + 5  # Cluster 1
        neg = np.random.randn(50, 32) - 5  # Cluster 2
        sep = stats.separability(pos, neg)
        assert sep > 0.9, "Well-separated clusters should have high AUC"

    def test_separability_overlapping(self, stats):
        """Overlapping clusters have ~0.5 AUC."""
        data = np.random.randn(100, 32)
        pos = data[:50]
        neg = data[50:]
        sep = stats.separability(pos, neg)
        assert 0.3 < sep < 0.7, "Overlapping clusters should have ~0.5 AUC"

    def test_diagnose_failure(self, stats):
        """Diagnosis provides useful feedback."""
        bad_metrics = {
            MetricName.STABILITY: 0.3,
            MetricName.SEPARABILITY: 0.4,
            MetricName.MUTUAL_INFORMATION: 0.1,
            MetricName.KL_DRIFT: 0.8,
        }
        issues = stats.diagnose_failure(bad_metrics)
        assert len(issues) >= 3, "Should identify multiple issues"


class TestOrchestrator:
    """Test the main orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        return Orchestrator(OrchestratorConfig(seed=42))

    def test_evaluate_returns_result(self, orchestrator):
        """Basic evaluation works."""
        result = orchestrator.evaluate(
            claim="Test claim",
            positives=["Example 1", "Example 2", "Example 3"],
            hypothesis_type=HypothesisType.CONCEPT_INVARIANT,
        )
        assert result.outcome in EvaluationOutcome

    def test_progress_callback(self, orchestrator):
        """Progress callbacks fire."""
        events = []

        def on_progress(event):
            events.append(event)

        orchestrator.evaluate(
            claim="Test",
            positives=["A", "B", "C"],
            hypothesis_type=HypothesisType.CONCEPT_INVARIANT,
            on_progress=on_progress,
        )

        assert len(events) > 0, "Should have progress events"
        step_types = [e.step for e in events]
        assert EvalStep.HYPOTHESIS_GEN in step_types
        assert EvalStep.NEGATIVE_GEN in step_types
        assert EvalStep.COMPUTE_METRICS in step_types

    def test_reproducibility(self, orchestrator):
        """Same seed gives same result."""
        args = dict(
            claim="Reproducible test",
            positives=["X", "Y", "Z"],
            hypothesis_type=HypothesisType.CONCEPT_INVARIANT,
        )

        orch1 = Orchestrator(OrchestratorConfig(seed=42))
        orch2 = Orchestrator(OrchestratorConfig(seed=42))

        r1 = orch1.evaluate(**args)
        r2 = orch2.evaluate(**args)

        assert r1.outcome == r2.outcome


class TestProviders:
    """Test provider abstractions."""

    def test_stub_llm_generates_hypothesis(self):
        """Stub LLM produces valid hypothesis."""
        from invariant.models.schemas import ModelReference

        config = ProviderConfig(model_ref=ModelReference(provider="stub", model_id="test"))
        llm = StubLLMProvider(config)

        hyp = llm.generate_hypothesis("Test", HypothesisType.CONCEPT_INVARIANT)
        assert hyp.title == "Test"
        assert hyp.hypothesis_type == HypothesisType.CONCEPT_INVARIANT

    def test_stub_llm_generates_negatives(self):
        """Stub LLM produces negatives."""
        from invariant.models.schemas import ModelReference

        config = ProviderConfig(model_ref=ModelReference(provider="stub", model_id="test"))
        llm = StubLLMProvider(config)

        negs = llm.generate_negatives(["A", "B"], NegativeStrategy.NEAR_MISS, n_per_positive=2)
        assert len(negs) == 4  # 2 per positive

    def test_stub_embedding_dimensions(self):
        """Stub embeddings have correct shape."""
        from invariant.models.schemas import ModelReference

        config = ProviderConfig(
            model_ref=ModelReference(provider="stub", model_id="test", embedding_dim=64)
        )
        emb = StubEmbeddingProvider(config)

        vectors = emb.embed(["text1", "text2", "text3"])
        assert vectors.shape == (3, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
