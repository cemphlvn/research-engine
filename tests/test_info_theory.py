"""Tests for information-theoretic primitives."""

import numpy as np
import pytest

from invariant.utils.info_theory import (
    entropy,
    kl_divergence,
    mutual_information,
    stability_score,
    info_preserved_ratio,
)


class TestEntropy:
    def test_uniform_distribution(self):
        # Uniform distribution should have high entropy
        x = np.random.uniform(0, 1, 1000)
        h = entropy(x)
        assert h > 0

    def test_constant_low_entropy(self):
        # Near-constant should have low entropy
        x = np.ones(1000) + np.random.normal(0, 0.01, 1000)
        h = entropy(x)
        # Still positive but lower than uniform
        assert h >= 0


class TestKLDivergence:
    def test_same_distribution(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        kl = kl_divergence(p, p)
        assert kl < 0.01  # Should be ~0

    def test_different_distributions(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        kl = kl_divergence(p, q)
        assert kl > 0  # Should be positive

    def test_non_negative(self):
        # KL divergence is always non-negative
        for _ in range(10):
            p = np.random.dirichlet(np.ones(5))
            q = np.random.dirichlet(np.ones(5))
            assert kl_divergence(p, q) >= 0


class TestMutualInformation:
    def test_independent_low_mi(self):
        x = np.random.randn(100)
        y = np.random.randn(100)
        mi = mutual_information(x, y, method="binned")
        # Independent variables should have low MI
        assert mi < 1.0

    def test_identical_high_mi(self):
        x = np.random.randn(100)
        mi = mutual_information(x, x, method="binned")
        # Identical variables should have high MI
        assert mi > 0


class TestStabilityScore:
    def test_identical_embeddings(self):
        emb = np.random.randn(10, 128)
        score = stability_score(emb, emb)
        assert score > 0.99

    def test_orthogonal_embeddings(self):
        orig = np.random.randn(10, 128)
        # Create orthogonal by negating
        trans = -orig
        score = stability_score(orig, trans)
        # Should be low (around 0 when mapped to [0,1])
        assert score < 0.1

    def test_similar_embeddings(self):
        orig = np.random.randn(10, 128)
        # Add small noise
        trans = orig + np.random.randn(10, 128) * 0.1
        score = stability_score(orig, trans)
        assert score > 0.9


class TestInfoPreservedRatio:
    def test_identical_full_preservation(self):
        emb = np.random.randn(50, 32)
        ratio = info_preserved_ratio(emb, emb)
        # Should be high (ideally 1.0)
        assert ratio > 0.5

    def test_random_low_preservation(self):
        orig = np.random.randn(50, 32)
        trans = np.random.randn(50, 32)
        ratio = info_preserved_ratio(orig, trans)
        # Random should have low preservation
        assert ratio < 0.5
