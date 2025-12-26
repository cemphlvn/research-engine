"""OpenAI embedding provider."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

from invariant.providers.base import EmbeddingProvider, ProviderConfig
from invariant.models.schemas import ModelReference


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3-small."""

    def __init__(self, config: ProviderConfig | None = None):
        if config is None:
            config = ProviderConfig(
                model_ref=ModelReference(
                    provider="openai",
                    model_id="text-embedding-3-small",
                    embedding_dim=1536,
                )
            )
        super().__init__(config)

        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        self._dimension = config.model_ref.embedding_dim or 1536

    def embed(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> NDArray[np.float64]:
        """Embed texts using OpenAI API."""
        if not texts:
            return np.array([])

        response = self.client.embeddings.create(
            model=self.config.model_ref.model_id,
            input=texts,
        )

        embeddings = np.array([d.embedding for d in response.data], dtype=np.float64)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)

        return embeddings

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
        normalize: bool = True,
    ) -> NDArray[np.float64]:
        """Batch embedding for large inputs."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embed(batch, normalize=normalize)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings) if all_embeddings else np.array([])
