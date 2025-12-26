"""Base provider abstractions.

Architecture:
- LLM = hypothesis generator + transformation generator + adversary
- Embedding = measurement instrument
- Stats = epistemic court (separate module)
- Orchestrator = scientific method (uses all above)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from invariant.models.schemas import (
    Hypothesis,
    HypothesisType,
    ClaimTarget,
    NegativeStrategy,
    Transform,
    TransformCategory,
    ModelReference,
)


@dataclass
class ProviderConfig:
    """Configuration for a provider."""

    model_ref: ModelReference
    timeout: float = 30.0
    max_retries: int = 3
    cache_enabled: bool = True


# =============================================================================
# LLM PROVIDER: hypothesis gen + transform gen + adversary
# =============================================================================


class LLMProvider(ABC):
    """Abstract LLM provider.

    Roles:
    1. Hypothesis generator - create candidate invariants from claims
    2. Transformation generator - generate paraphrases, rewrites
    3. Adversary - generate near-miss negatives
    """

    def __init__(self, config: ProviderConfig):
        self.config = config

    # --- Hypothesis Generation ---

    @abstractmethod
    def generate_hypothesis(
        self,
        claim: str,
        hypothesis_type: HypothesisType,
        context: str | None = None,
    ) -> Hypothesis:
        """Generate a structured hypothesis from a claim."""
        ...

    @abstractmethod
    def extract_predicted_invariants(
        self,
        claim: str,
    ) -> list[str]:
        """Extract what should remain invariant if claim is true."""
        ...

    @abstractmethod
    def formalize_hypothesis(
        self,
        hypothesis: Hypothesis,
    ) -> str | dict[str, Any]:
        """Convert hypothesis to formal representation."""
        ...

    # --- Transformation Generation ---

    @abstractmethod
    def paraphrase(
        self,
        text: str,
        n_variants: int = 1,
        preserve: list[str] | None = None,
    ) -> list[str]:
        """Generate paraphrases preserving specified invariants."""
        ...

    @abstractmethod
    def apply_transform(
        self,
        text: str,
        transform: Transform,
    ) -> str:
        """Apply a semantic transformation."""
        ...

    @abstractmethod
    def generate_transform_variants(
        self,
        base_transform: Transform,
        n_variants: int = 3,
    ) -> list[Transform]:
        """Generate variations of a transform."""
        ...

    # --- Adversary (Negative Generation) ---

    @abstractmethod
    def generate_negatives(
        self,
        positives: list[str],
        strategy: NegativeStrategy,
        n_per_positive: int = 2,
    ) -> list[str]:
        """Generate negative examples for falsifiability."""
        ...

    @abstractmethod
    def generate_near_miss(
        self,
        positive: str,
        target_invariant: str,
    ) -> str:
        """Generate adversarial near-miss that should fail the invariant."""
        ...

    # --- Utility ---

    def generate_negatives_batch(
        self,
        positives: list[str],
        strategy: NegativeStrategy = NegativeStrategy.NEAR_MISS,
        n_per_positive: int = 2,
    ) -> list[str]:
        """Batch negative generation (default calls single)."""
        return self.generate_negatives(positives, strategy, n_per_positive)


# =============================================================================
# EMBEDDING PROVIDER: measurement instrument
# =============================================================================


class EmbeddingProvider(ABC):
    """Abstract embedding provider.

    Role: Measurement instrument
    - Provides the geometric substrate for invariance metrics
    - Invariance is always measured through this lens
    - Different embeddings = different invariants
    """

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        if self._dimension is None:
            # Probe with empty call
            test = self.embed(["test"])
            self._dimension = test.shape[1]
        return self._dimension

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> NDArray[np.float64]:
        """Embed texts to vectors.

        Args:
            texts: List of texts to embed
            normalize: Whether to L2 normalize

        Returns:
            (N, D) array of embeddings
        """
        ...

    @abstractmethod
    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> NDArray[np.float64]:
        """Batch embedding for large inputs."""
        ...

    def similarity(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
    ) -> float:
        """Compute cosine similarity between embeddings."""
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a_norm, b_norm))


# =============================================================================
# STUB IMPLEMENTATIONS (for testing without API)
# =============================================================================


class StubLLMProvider(LLMProvider):
    """Stub LLM for testing - uses simple heuristics."""

    def generate_hypothesis(
        self,
        claim: str,
        hypothesis_type: HypothesisType,
        context: str | None = None,
    ) -> Hypothesis:
        return Hypothesis(
            title=claim,
            target=ClaimTarget(mode="text", content=claim),
            hypothesis_type=hypothesis_type,
            predicted_invariants=[f"Semantic content of: {claim}"],
        )

    def extract_predicted_invariants(self, claim: str) -> list[str]:
        return [f"Core meaning: {claim[:50]}"]

    def formalize_hypothesis(self, hypothesis: Hypothesis) -> str:
        return f"FORMAL: {hypothesis.title}"

    def paraphrase(
        self,
        text: str,
        n_variants: int = 1,
        preserve: list[str] | None = None,
    ) -> list[str]:
        # Simple word reordering
        words = text.split()
        variants = []
        for i in range(n_variants):
            import random
            shuffled = words.copy()
            # Only shuffle non-critical words
            if len(shuffled) > 3:
                mid = shuffled[1:-1]
                random.shuffle(mid)
                shuffled = [shuffled[0]] + mid + [shuffled[-1]]
            variants.append(" ".join(shuffled))
        return variants

    def apply_transform(self, text: str, transform: Transform) -> str:
        """Apply transform (stub uses simple heuristics)."""
        name = transform.name.lower()

        # Representation transforms
        if "paraphrase" in name:
            # Simple word reordering as stub paraphrase
            words = text.split()
            if len(words) > 3:
                mid = len(words) // 2
                words[1], words[mid] = words[mid], words[1]
            return " ".join(words)
        elif "formalize" in name:
            return f"It is the case that {text.lower()}"
        elif "simplify" in name:
            return text.lower()
        elif "lowercase" in name:
            return text.lower()

        # Context transforms
        elif "context" in name or "add" in name:
            return f"Consider that {text.lower()}"
        elif "abstract" in name:
            return f"In general, {text.lower()}"
        elif "concrete" in name:
            return f"Specifically, {text}"

        return text

    def generate_transform_variants(
        self,
        base_transform: Transform,
        n_variants: int = 3,
    ) -> list[Transform]:
        return [base_transform]  # No variants in stub

    def generate_negatives(
        self,
        positives: list[str],
        strategy: NegativeStrategy,
        n_per_positive: int = 2,
    ) -> list[str]:
        """Generate adversarial near-miss negatives.

        These should be textually similar but semantically different to test
        that the embedding captures meaning, not just lexical overlap.
        """
        import random
        random.seed(42)

        # Near-synonyms that change meaning subtly
        subtle_swaps = {
            "justice": "justification", "fair": "familiar", "equal": "equate",
            "good": "goods", "right": "rite", "true": "truth",
            "love": "lovely", "happy": "happen", "success": "succession",
            "all": "most", "always": "often", "never": "rarely",
            "must": "might", "is": "was", "are": "were",
            "will": "would", "can": "could", "should": "might",
        }

        negatives = []
        for p in positives:
            words = p.split()

            # Strategy 1: Subtle word swap (preserves structure)
            swapped = []
            changed = False
            for w in words:
                w_lower = w.lower()
                if w_lower in subtle_swaps and not changed:
                    swapped.append(subtle_swaps[w_lower])
                    changed = True
                else:
                    swapped.append(w)
            if changed:
                negatives.append(" ".join(swapped))
            else:
                # If no swap found, add qualifier that changes meaning
                negatives.append(f"Sometimes {p.lower()}")

            # Strategy 2: Word order change (similar chars, different meaning)
            if len(words) > 2:
                reordered = words.copy()
                mid = len(reordered) // 2
                reordered[0], reordered[mid] = reordered[mid], reordered[0]
                negatives.append(" ".join(reordered))
            else:
                negatives.append(f"Perhaps {p.lower()}")

        # Trim to requested count
        return negatives[:len(positives) * n_per_positive]

    def generate_near_miss(self, positive: str, target_invariant: str) -> str:
        """Generate adversarial near-miss using negation."""
        return f"It is not the case that {positive.lower()}"


class StubEmbeddingProvider(EmbeddingProvider):
    """Stub embeddings that preserve semantic similarity for testing.

    Uses character n-gram + word overlap features so similar texts
    produce similar vectors. This makes the system testable without APIs.
    """

    def __init__(self, config: ProviderConfig, dimension: int | None = None):
        super().__init__(config)
        self._dimension = config.model_ref.embedding_dim or dimension or 128
        # Fixed random projection matrix for consistent dimensionality reduction
        np.random.seed(42)
        self._projection = np.random.randn(1000, self._dimension) / np.sqrt(self._dimension)

    def _text_to_features(self, text: str) -> np.ndarray:
        """Convert text to sparse feature vector preserving similarity."""
        text = text.lower().strip()
        features = np.zeros(1000)

        # Character 3-grams (indices 0-500)
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            idx = hash(trigram) % 500
            features[idx] += 1

        # Word features (indices 500-800)
        words = text.split()
        for word in words:
            idx = 500 + (hash(word) % 300)
            features[idx] += 1

        # Word bigrams (indices 800-1000)
        for i in range(len(words) - 1):
            bigram = words[i] + " " + words[i+1]
            idx = 800 + (hash(bigram) % 200)
            features[idx] += 1

        return features

    def embed(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> NDArray[np.float64]:
        embeddings = []
        for text in texts:
            features = self._text_to_features(text)
            # Project to lower dimension
            vec = features @ self._projection
            embeddings.append(vec)

        result = np.array(embeddings, dtype=np.float64)

        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / (norms + 1e-10)

        return result

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> NDArray[np.float64]:
        return self.embed(texts, normalize)
