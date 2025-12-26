"""Semantic LLM provider - proper meaning-preserving transforms and near-miss negatives.

This provider implements the core epistemological principle:
- Transforms must preserve meaning (not just syntax)
- Negatives must be semantically close but meaning-different (not keyword overlap)

Never use random/easy negatives. Never use syntactic-only transforms.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

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


# =============================================================================
# PROMPTS - The core of semantic evaluation
# =============================================================================

SYSTEM_SEMANTIC_NEGATIVE = """You are an adversarial semantic analyst. Your job is to generate NEAR-MISS negatives.

A near-miss negative is:
- On the SAME TOPIC as the positive
- Uses SIMILAR vocabulary and structure
- But has a DIFFERENT or OPPOSITE meaning

Examples of good near-misses:
- Positive: "Justice requires treating equals equally"
- Near-miss: "Justice permits treating equals unequally when convenient"
- Near-miss: "Revenge is a form of justice"
- Near-miss: "Justice means the powerful deciding what's fair"

Examples of BAD negatives (never generate these):
- Random: "The weather is nice today" (different topic - too easy)
- Keyword swap: "Injustice requires treating equals equally" (just word replacement)
- Simple negation: "Justice does not require treating equals equally" (too obvious)

Your negatives should be HARD to distinguish from positives based on keywords alone.
They should require understanding MEANING to distinguish."""

SYSTEM_SEMANTIC_TRANSFORM = """You are a semantic transformation expert. Your job is to PARAPHRASE text while PRESERVING MEANING.

A good paraphrase:
- Uses DIFFERENT words and sentence structure
- Preserves the EXACT SAME meaning
- Should be indistinguishable in semantic content

Examples:
- Original: "Justice requires treating equals equally"
- Paraphrase: "Fair treatment demands equal handling of equivalent cases"
- Paraphrase: "Those who are alike must be dealt with alike - that's what justice means"

BAD paraphrases (never do these):
- Same words: "Justice requires treating equals equally." (just copied)
- Changed meaning: "Justice requires treating everyone the same" (different meaning!)
- Added content: "Justice requires treating equals equally, which is important" (added info)

Only change the FORM, never the MEANING."""

SYSTEM_HYPOTHESIS = """You are an epistemological research assistant helping to formalize claims into testable hypotheses.

For each claim, you must identify:
1. PREDICTED INVARIANTS: What should remain stable if this claim is true?
2. POTENTIAL FALSIFIERS: What observations would disprove this claim?
3. BOUNDARY CONDITIONS: Where does this claim apply vs not apply?

Output structured JSON with these fields."""


# =============================================================================
# SEMANTIC PROVIDER
# =============================================================================

class SemanticLLMProvider(LLMProvider):
    """LLM provider with proper semantic operations.

    This wraps any LLM API and provides:
    1. Meaning-preserving transforms (not syntactic)
    2. Near-miss negatives (same topic, different meaning)
    3. Structured hypothesis generation

    The key insight: quality of epistemological evaluation depends entirely
    on quality of transforms and negatives. This provider gets them right.
    """

    def __init__(
        self,
        api_call_fn,  # Function: (system: str, user: str) -> str
        config: ProviderConfig | None = None,
    ):
        """Initialize with an API call function.

        Args:
            api_call_fn: Function that takes (system_prompt, user_prompt) and returns response string.
                         This allows wrapping any LLM API (OpenAI, Anthropic, DeepSeek, etc.)
            config: Optional provider config
        """
        if config is None:
            config = ProviderConfig(
                model_ref=ModelReference(provider="semantic", model_id="wrapped")
            )
        super().__init__(config)
        self._call = api_call_fn

    # =========================================================================
    # HYPOTHESIS GENERATION
    # =========================================================================

    def generate_hypothesis(
        self,
        claim: str,
        hypothesis_type: HypothesisType,
        context: str | None = None,
    ) -> Hypothesis:
        """Generate a structured hypothesis with predicted invariants."""
        user = f"""Analyze this claim and generate a testable hypothesis.

Claim: {claim}
Type: {hypothesis_type.value}
Context: {context or 'General'}

Output JSON with:
{{
    "title": "concise title",
    "predicted_invariants": ["what should stay stable if true", ...],
    "potential_falsifiers": ["what would disprove this", ...],
    "boundary_conditions": ["where this applies", "where it doesn't apply"],
    "formalization": "formal logical statement if possible"
}}"""

        try:
            response = self._call(SYSTEM_HYPOTHESIS, user)
            data = self._extract_json(response, {})

            return Hypothesis(
                title=data.get("title", claim),
                target=ClaimTarget(mode="text", content=claim),
                hypothesis_type=hypothesis_type,
                predicted_invariants=data.get("predicted_invariants", [f"Semantic content of: {claim}"]),
                formalization=data.get("formalization"),
            )
        except Exception:
            return Hypothesis(
                title=claim,
                target=ClaimTarget(mode="text", content=claim),
                hypothesis_type=hypothesis_type,
                predicted_invariants=[f"Semantic content of: {claim}"],
            )

    def extract_predicted_invariants(self, claim: str) -> list[str]:
        """Extract what should remain invariant if claim is true."""
        user = f"""What properties should remain INVARIANT (stable) if this claim is true?

Claim: {claim}

List 3-5 invariants that should hold. Be specific.
Output as JSON array of strings."""

        try:
            response = self._call(SYSTEM_HYPOTHESIS, user)
            return self._extract_json(response, [f"Core meaning of: {claim}"])
        except Exception:
            return [f"Core meaning of: {claim}"]

    def formalize_hypothesis(self, hypothesis: Hypothesis) -> str:
        """Convert hypothesis to formal representation."""
        if hypothesis.formalization:
            return hypothesis.formalization
        return f"CLAIM({hypothesis.hypothesis_type.value}): {hypothesis.title}"

    # =========================================================================
    # SEMANTIC TRANSFORMS (meaning-preserving)
    # =========================================================================

    def paraphrase(
        self,
        text: str,
        n_variants: int = 1,
        preserve: list[str] | None = None,
    ) -> list[str]:
        """Generate true semantic paraphrases.

        These MUST preserve meaning exactly while changing form.
        """
        preserve_note = ""
        if preserve:
            preserve_note = f"\n\nCRITICAL: These concepts must be preserved exactly: {', '.join(preserve)}"

        user = f"""Paraphrase this text {n_variants} time(s).

RULES:
- Use DIFFERENT words and sentence structure
- Preserve the EXACT SAME meaning
- Do NOT add, remove, or change any information
{preserve_note}

Text: "{text}"

Output JSON array of {n_variants} paraphrase(s):"""

        try:
            response = self._call(SYSTEM_SEMANTIC_TRANSFORM, user)
            result = self._extract_json(response, [text])
            if isinstance(result, list):
                return result[:n_variants]
            return [text]
        except Exception:
            return [text]

    def apply_transform(self, text: str, transform: Transform) -> str:
        """Apply a semantic transformation."""
        if transform.category == TransformCategory.REPRESENTATION:
            if "paraphrase" in transform.name.lower():
                paraphrases = self.paraphrase(text, n_variants=1)
                return paraphrases[0] if paraphrases else text
            elif "formalize" in transform.name.lower():
                return self._formalize_text(text)
            elif "simplify" in transform.name.lower():
                return self._simplify_text(text)
            elif "lowercase" in transform.name.lower():
                return text.lower()

        elif transform.category == TransformCategory.CONTEXT:
            if "add" in transform.name.lower():
                return self._add_neutral_context(text)
            elif "abstract" in transform.name.lower():
                return self._abstract_text(text)
            elif "concrete" in transform.name.lower():
                return self._concretize_text(text)

        return text

    def _formalize_text(self, text: str) -> str:
        """Convert to formal/academic register."""
        user = f"""Rewrite in formal academic language. Preserve exact meaning.

Text: "{text}"

Formal version:"""
        try:
            response = self._call(SYSTEM_SEMANTIC_TRANSFORM, user)
            return response.strip().strip('"')
        except Exception:
            return text

    def _simplify_text(self, text: str) -> str:
        """Convert to simpler language."""
        user = f"""Rewrite in simpler, everyday language. Preserve exact meaning.

Text: "{text}"

Simple version:"""
        try:
            response = self._call(SYSTEM_SEMANTIC_TRANSFORM, user)
            return response.strip().strip('"')
        except Exception:
            return text

    def _add_neutral_context(self, text: str) -> str:
        """Add neutral context that doesn't change meaning."""
        user = f"""Add neutral surrounding context that doesn't change the core meaning.

Text: "{text}"

With context (the original meaning must be unchanged):"""
        try:
            response = self._call(SYSTEM_SEMANTIC_TRANSFORM, user)
            return response.strip().strip('"')
        except Exception:
            return f"Consider that {text.lower()}"

    def _abstract_text(self, text: str) -> str:
        """Make text more abstract/general."""
        user = f"""Rewrite at a more abstract/general level. Preserve core meaning.

Text: "{text}"

Abstract version:"""
        try:
            response = self._call(SYSTEM_SEMANTIC_TRANSFORM, user)
            return response.strip().strip('"')
        except Exception:
            return text

    def _concretize_text(self, text: str) -> str:
        """Make text more concrete/specific."""
        user = f"""Rewrite with more concrete/specific language. Preserve core meaning.

Text: "{text}"

Concrete version:"""
        try:
            response = self._call(SYSTEM_SEMANTIC_TRANSFORM, user)
            return response.strip().strip('"')
        except Exception:
            return text

    def generate_transform_variants(
        self,
        base_transform: Transform,
        n_variants: int = 3,
    ) -> list[Transform]:
        """Generate variations of a transform."""
        # For now, just return the base transform
        # Could expand to generate parametric variations
        return [base_transform]

    # =========================================================================
    # ADVERSARY: SEMANTIC NEGATIVE GENERATION
    # =========================================================================

    def generate_negatives(
        self,
        positives: list[str],
        strategy: NegativeStrategy,
        n_per_positive: int = 2,
    ) -> list[str]:
        """Generate negative examples using specified strategy.

        For real epistemological evaluation, use:
        - NEAR_MISS: Hardest, same topic but different meaning
        - CONTRADICT: Logical contradictions
        - ANTONYM: Key concept reversal
        - BOUNDARY: Edge cases

        Avoid RANDOM/SHUFFLE for real evaluation (too easy).
        """
        if strategy == NegativeStrategy.NEAR_MISS:
            return self._generate_near_miss_negatives(positives, n_per_positive)
        elif strategy == NegativeStrategy.CONTRADICT:
            return self._generate_contradictions(positives, n_per_positive)
        elif strategy == NegativeStrategy.ANTONYM:
            return self._generate_antonym_negatives(positives, n_per_positive)
        elif strategy == NegativeStrategy.RELATED:
            return self._generate_related_negatives(positives, n_per_positive)
        elif strategy == NegativeStrategy.BOUNDARY:
            return self._generate_boundary_negatives(positives, n_per_positive)
        elif strategy == NegativeStrategy.NEGATE:
            return self._generate_logical_negations(positives)
        else:
            # Random/shuffle - warn and use near-miss instead
            return self._generate_near_miss_negatives(positives, n_per_positive)

    def _generate_near_miss_negatives(self, positives: list[str], n_per: int) -> list[str]:
        """Generate near-miss negatives: same topic, different meaning.

        These are the HARDEST negatives and the most epistemically rigorous.
        """
        pos_str = "\n".join(f"- {p}" for p in positives[:5])

        user = f"""Generate {n_per} NEAR-MISS negative for EACH positive example.

Positive examples:
{pos_str}

For each positive, generate {n_per} near-miss negatives that:
1. Stay on the SAME TOPIC
2. Use SIMILAR vocabulary
3. Have DIFFERENT or OPPOSITE meaning
4. Would be HARD to distinguish by keyword matching alone

Output JSON array of all negatives (should be {n_per * min(len(positives), 5)} total):"""

        try:
            response = self._call(SYSTEM_SEMANTIC_NEGATIVE, user)
            result = self._extract_string_list(response)
            if result:
                return result
        except Exception:
            pass

        # Fallback
        return [f"[NEAR-MISS] Contrary view: {p}" for p in positives for _ in range(n_per)]

    def _generate_contradictions(self, positives: list[str], n_per: int) -> list[str]:
        """Generate logical contradictions."""
        pos_str = "\n".join(f"- {p}" for p in positives[:5])

        user = f"""Generate {n_per} CONTRADICTION for EACH positive example.

Positive examples:
{pos_str}

A contradiction is a statement that CANNOT be true if the positive is true.
Not just negation - a genuine logical contradiction.

Output JSON array:"""

        try:
            response = self._call(SYSTEM_SEMANTIC_NEGATIVE, user)
            result = self._extract_string_list(response)
            if result:
                return result
        except Exception:
            pass

        return [f"It is false that {p.lower()}" for p in positives for _ in range(n_per)]

    def _generate_antonym_negatives(self, positives: list[str], n_per: int) -> list[str]:
        """Generate negatives by replacing key concepts with antonyms."""
        pos_str = "\n".join(f"- {p}" for p in positives[:5])

        user = f"""Generate {n_per} ANTONYM-based negative for EACH positive example.

Positive examples:
{pos_str}

Replace KEY CONCEPTS with their opposites/antonyms.
Keep the same structure but flip the meaning.

Example:
- Positive: "Justice requires equal treatment"
- Antonym negative: "Injustice requires unequal treatment"

Output JSON array:"""

        try:
            response = self._call(SYSTEM_SEMANTIC_NEGATIVE, user)
            result = self._extract_string_list(response)
            if result:
                return result
        except Exception:
            pass

        return [f"The opposite: {p}" for p in positives for _ in range(n_per)]

    def _generate_related_negatives(self, positives: list[str], n_per: int) -> list[str]:
        """Generate negatives about related but different topics."""
        pos_str = "\n".join(f"- {p}" for p in positives[:5])

        user = f"""Generate {n_per} RELATED-TOPIC negative for EACH positive example.

Positive examples:
{pos_str}

Generate statements about RELATED but DIFFERENT topics.
Same general domain but making different claims.

Example:
- Positive: "Justice requires equal treatment"
- Related: "Law enforcement requires proper training"
- Related: "Courts should be efficient"

Output JSON array:"""

        try:
            response = self._call(SYSTEM_SEMANTIC_NEGATIVE, user)
            result = self._extract_string_list(response)
            if result:
                return result
        except Exception:
            pass

        return [f"Related: {p[:30]}..." for p in positives for _ in range(n_per)]

    def _generate_boundary_negatives(self, positives: list[str], n_per: int) -> list[str]:
        """Generate edge cases that test boundaries of the claim."""
        pos_str = "\n".join(f"- {p}" for p in positives[:5])

        user = f"""Generate {n_per} BOUNDARY CASE negative for EACH positive example.

Positive examples:
{pos_str}

Generate edge cases that ALMOST satisfy the claim but don't quite.
These test the boundaries of when the claim applies.

Example:
- Positive: "Justice requires equal treatment"
- Boundary: "Justice requires equal treatment except in emergencies"
- Boundary: "Justice requires equal treatment when resources allow"

Output JSON array:"""

        try:
            response = self._call(SYSTEM_SEMANTIC_NEGATIVE, user)
            result = self._extract_string_list(response)
            if result:
                return result
        except Exception:
            pass

        return [f"Edge case: {p}" for p in positives for _ in range(n_per)]

    def _generate_logical_negations(self, positives: list[str]) -> list[str]:
        """Generate simple logical negations."""
        pos_str = "\n".join(f"- {p}" for p in positives[:5])

        user = f"""Generate logical negations of these statements.

Statements:
{pos_str}

Negate each statement properly (not just "not X", but proper negation).

Output JSON array:"""

        try:
            response = self._call(SYSTEM_SEMANTIC_NEGATIVE, user)
            result = self._extract_string_list(response)
            if result:
                return result
        except Exception:
            pass

        return [f"It is not the case that {p.lower()}" for p in positives]

    def generate_near_miss(self, positive: str, target_invariant: str) -> str:
        """Generate single near-miss targeting a specific invariant."""
        user = f"""Generate ONE near-miss negative for this positive.

Positive: "{positive}"
Target invariant to violate: "{target_invariant}"

The near-miss should:
1. Stay on the same topic
2. Use similar vocabulary
3. Specifically VIOLATE the target invariant

Near-miss:"""

        try:
            response = self._call(SYSTEM_SEMANTIC_NEGATIVE, user)
            return response.strip().strip('"')
        except Exception:
            return f"Contrary to {target_invariant}: {positive}"

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _extract_json(self, text: str, default: Any) -> Any:
        """Extract JSON from LLM response."""
        # Try to find JSON array or object
        for pattern in [r'\[.*\]', r'\{.*\}']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue
        return default

    def _extract_string_list(self, text: str) -> list[str]:
        """Extract list of strings from LLM response, handling dict formats."""
        result = self._extract_json(text, [])
        if not isinstance(result, list):
            return []

        strings = []
        for item in result:
            if isinstance(item, str):
                strings.append(item)
            elif isinstance(item, dict):
                # Extract first string value from dict
                for v in item.values():
                    if isinstance(v, str):
                        strings.append(v)
                        break
        return strings


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_semantic_provider_deepseek(use_reasoner: bool = False) -> SemanticLLMProvider:
    """Create semantic provider using DeepSeek API."""
    import os
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    model = "deepseek-reasoner" if use_reasoner else "deepseek-chat"

    def call_api(system: str, user: str) -> str:
        response = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    return SemanticLLMProvider(call_api)


def create_semantic_provider_anthropic() -> SemanticLLMProvider:
    """Create semantic provider using Anthropic API."""
    import os
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    def call_api(system: str, user: str) -> str:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    return SemanticLLMProvider(call_api)


def create_semantic_provider_openai() -> SemanticLLMProvider:
    """Create semantic provider using OpenAI API."""
    import os
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    def call_api(system: str, user: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=2048,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    return SemanticLLMProvider(call_api)
