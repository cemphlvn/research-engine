"""Provider abstractions for LLM, embedding, and stats layers."""

from invariant.providers.base import (
    LLMProvider,
    EmbeddingProvider,
    ProviderConfig,
    StubLLMProvider,
    StubEmbeddingProvider,
)
from invariant.providers.stats import StatsEngine, StatsConfig
from invariant.providers.semantic import (
    SemanticLLMProvider,
    create_semantic_provider_deepseek,
    create_semantic_provider_anthropic,
    create_semantic_provider_openai,
)

__all__ = [
    # Base abstractions
    "LLMProvider",
    "EmbeddingProvider",
    "ProviderConfig",
    # Stubs for testing
    "StubLLMProvider",
    "StubEmbeddingProvider",
    # Stats
    "StatsEngine",
    "StatsConfig",
    # Semantic (preferred for real evaluation)
    "SemanticLLMProvider",
    "create_semantic_provider_deepseek",
    "create_semantic_provider_anthropic",
    "create_semantic_provider_openai",
]
