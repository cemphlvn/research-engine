"""DeepSeek LLM provider - best price/reasoning ratio Dec 2025.

DeepSeek V3.2: $0.28/M input (94% cheaper than Claude)
DeepSeek R1: $0.55/M input (27x cheaper than o1)

Uses OpenAI-compatible API.
"""

from __future__ import annotations

import os
import json
from typing import Any

from invariant.providers.base import LLMProvider, ProviderConfig
from invariant.models.schemas import (
    Hypothesis,
    HypothesisType,
    ClaimTarget,
    NegativeStrategy,
    Transform,
    TransformCategory,
    ModelReference,
)


class DeepSeekLLMProvider(LLMProvider):
    """DeepSeek provider - best price/performance for reasoning.

    Models:
    - deepseek-chat: General tasks ($0.28/M input)
    - deepseek-reasoner: R1 with CoT reasoning ($0.55/M input)
    """

    def __init__(self, config: ProviderConfig | None = None, use_reasoner: bool = False):
        model_id = "deepseek-reasoner" if use_reasoner else "deepseek-chat"
        if config is None:
            config = ProviderConfig(
                model_ref=ModelReference(
                    provider="deepseek",
                    model_id=model_id,
                )
            )
        super().__init__(config)
        self.use_reasoner = use_reasoner

        from openai import OpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def _call(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Make a DeepSeek API call."""
        response = self.client.chat.completions.create(
            model=self.config.model_ref.model_id,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    # --- Hypothesis Generation ---

    def generate_hypothesis(
        self,
        claim: str,
        hypothesis_type: HypothesisType,
        context: str | None = None,
    ) -> Hypothesis:
        """Generate a structured hypothesis from a claim."""
        system = """You are an epistemological research assistant. Given a claim, generate a structured hypothesis.
Output JSON with: title, predicted_invariants (list of what should remain stable if true), formalization (formal statement)."""

        user = f"""Claim: {claim}
Type: {hypothesis_type.value}
Context: {context or 'None'}

Generate hypothesis as JSON."""

        try:
            response = self._call(system, user)
            if "{" in response:
                json_str = response[response.index("{"):response.rindex("}") + 1]
                data = json.loads(json_str)
                return Hypothesis(
                    title=data.get("title", claim),
                    target=ClaimTarget(mode="text", content=claim),
                    hypothesis_type=hypothesis_type,
                    predicted_invariants=data.get("predicted_invariants", []),
                    formalization=data.get("formalization"),
                )
        except Exception:
            pass

        return Hypothesis(
            title=claim,
            target=ClaimTarget(mode="text", content=claim),
            hypothesis_type=hypothesis_type,
            predicted_invariants=[f"Semantic content of: {claim}"],
        )

    def extract_predicted_invariants(self, claim: str) -> list[str]:
        """Extract what should remain invariant if claim is true."""
        system = "You extract predicted invariants from claims. Output a JSON list of strings."
        user = f"What should remain invariant if this claim is true?\n\nClaim: {claim}\n\nOutput JSON list:"

        try:
            response = self._call(system, user, max_tokens=512)
            if "[" in response:
                json_str = response[response.index("["):response.rindex("]") + 1]
                return json.loads(json_str)
        except Exception:
            pass

        return [f"Core meaning: {claim[:50]}"]

    def formalize_hypothesis(self, hypothesis: Hypothesis) -> str:
        """Convert hypothesis to formal representation."""
        return f"FORMAL({hypothesis.hypothesis_type.value}): {hypothesis.title}"

    # --- Transformation Generation ---

    def paraphrase(
        self,
        text: str,
        n_variants: int = 1,
        preserve: list[str] | None = None,
    ) -> list[str]:
        """Generate paraphrases preserving specified invariants."""
        preserve_str = ", ".join(preserve) if preserve else "core meaning"
        system = f"Generate paraphrases that preserve: {preserve_str}. Output JSON list of strings."
        user = f"Generate {n_variants} paraphrase(s) of:\n\n{text}\n\nOutput JSON list:"

        try:
            response = self._call(system, user, max_tokens=1024)
            if "[" in response:
                json_str = response[response.index("["):response.rindex("]") + 1]
                return json.loads(json_str)[:n_variants]
        except Exception:
            pass

        return [text]

    def apply_transform(self, text: str, transform: Transform) -> str:
        """Apply a semantic transformation."""
        if transform.category == TransformCategory.REPRESENTATION:
            if "paraphrase" in transform.name.lower():
                paraphrases = self.paraphrase(text, n_variants=1)
                return paraphrases[0] if paraphrases else text
            elif "lowercase" in transform.name.lower():
                return text.lower()
            else:
                return text

        elif transform.category == TransformCategory.CONTEXT:
            if "add" in transform.name.lower():
                return f"[Additional context] {text}"
            elif "remove" in transform.name.lower():
                if text.startswith("["):
                    end = text.find("]")
                    if end > 0:
                        return text[end + 1:].strip()
                return text

        return text

    def generate_transform_variants(
        self,
        base_transform: Transform,
        n_variants: int = 3,
    ) -> list[Transform]:
        """Generate variations of a transform."""
        return [base_transform]

    # --- Adversary (Negative Generation) ---

    def generate_negatives(
        self,
        positives: list[str],
        strategy: NegativeStrategy,
        n_per_positive: int = 2,
    ) -> list[str]:
        """Generate negative examples for falsifiability."""
        if strategy == NegativeStrategy.NEAR_MISS:
            return self._generate_near_misses(positives, n_per_positive)
        elif strategy == NegativeStrategy.NEGATE:
            return self._generate_negations(positives)
        else:
            import random
            negatives = []
            for p in positives:
                words = p.split()
                for _ in range(n_per_positive):
                    shuffled = words.copy()
                    random.shuffle(shuffled)
                    negatives.append(" ".join(shuffled))
            return negatives

    def _generate_near_misses(self, positives: list[str], n_per: int) -> list[str]:
        """Generate adversarial near-misses."""
        system = """Generate adversarial near-miss examples. These should be ALMOST correct but subtly wrong.
They should look similar to the positive examples but fail the invariant being tested.
Output JSON list of strings."""

        pos_str = "\n".join(f"- {p}" for p in positives[:5])
        user = f"""Positive examples:
{pos_str}

Generate {n_per * len(positives[:5])} near-miss negatives that are subtly wrong.
Output JSON list:"""

        try:
            response = self._call(system, user, max_tokens=1024)
            if "[" in response:
                json_str = response[response.index("["):response.rindex("]") + 1]
                return json.loads(json_str)
        except Exception:
            pass

        return [f"NOT: {p}" for p in positives for _ in range(n_per)]

    def _generate_negations(self, positives: list[str]) -> list[str]:
        """Generate logical negations."""
        system = "Generate logical negations. Output JSON list."
        pos_str = "\n".join(f"- {p}" for p in positives[:5])
        user = f"Negate each:\n{pos_str}\n\nOutput JSON list:"

        try:
            response = self._call(system, user, max_tokens=512)
            if "[" in response:
                json_str = response[response.index("["):response.rindex("]") + 1]
                return json.loads(json_str)
        except Exception:
            pass

        return [f"Not: {p}" for p in positives]

    def generate_near_miss(self, positive: str, target_invariant: str) -> str:
        """Generate single adversarial near-miss."""
        system = f"Generate ONE adversarial near-miss that violates: {target_invariant}"
        user = f"Positive: {positive}\n\nGenerate near-miss:"

        try:
            response = self._call(system, user, max_tokens=256)
            return response.strip()
        except Exception:
            return f"NOT: {positive}"
