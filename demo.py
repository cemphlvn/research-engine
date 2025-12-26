#!/usr/bin/env python3
"""Interactive demo of the Invariant engine."""

from invariant.core.orchestrator import Orchestrator, OrchestratorConfig
from invariant.models.schemas import HypothesisType, EvaluationOutcome

def run_demo(claim: str, positives: list[str], hypothesis_type: HypothesisType = HypothesisType.CONCEPT_INVARIANT):
    """Run the engine on a claim with positive examples."""
    print(f"\n{'='*60}")
    print(f"CLAIM: {claim}")
    print(f"TYPE: {hypothesis_type.value}")
    print(f"POSITIVES: {len(positives)} examples")
    print('='*60)

    # Initialize orchestrator (uses stubs by default)
    orchestrator = Orchestrator(OrchestratorConfig(seed=42))

    # Run evaluation
    print("\nRunning evaluation...")
    result = orchestrator.evaluate(
        claim=claim,
        positives=positives,
        hypothesis_type=hypothesis_type,
    )

    # Display result
    print(f"\n{'='*60}")
    print(f"OUTCOME: {result.outcome.value.upper()}")
    print('='*60)

    if result.outcome == EvaluationOutcome.DISCOVERY:
        d = result.discovery
        print(f"\n[DISCOVERY]")
        print(f"Confidence: {d.confidence:.0%}")
        print(f"\nFormal:\n{d.invariant_formal}")
        print(f"\nPlain: {d.invariant_plain}")
        if d.boundary_conditions:
            print(f"\nBoundaries:")
            for b in d.boundary_conditions[:3]:
                print(f"  - {b.description}")

    elif result.outcome == EvaluationOutcome.FALSIFIED:
        f = result.falsification
        print(f"\n[FALSIFIED]")
        print(f"Failure mode: {f.failure_mode.value}")
        print(f"Explanation: {f.explanation}")
        if f.counterexample:
            print(f"Breaking transform: {f.counterexample.breaking_transform}")
            print(f"Divergence: {f.counterexample.divergence_metric:.2f}")
        if f.suggested_refinements:
            print(f"\nSuggestions:")
            for s in f.suggested_refinements[:3]:
                print(f"  - {s}")

    elif result.outcome == EvaluationOutcome.UNDERDETERMINED:
        u = result.underdetermined
        print(f"\n[UNDERDETERMINED]")
        print(f"Reason: {u.reason}")
        if u.conflicting_signals:
            print(f"\nConflicting signals:")
            for c in u.conflicting_signals[:3]:
                print(f"  - {c}")
        if u.suggested_experiments:
            print(f"\nNext experiments:")
            for e in u.suggested_experiments[:3]:
                print(f"  - {e}")

    elif result.outcome == EvaluationOutcome.INADMISSIBLE:
        a = result.admissibility
        print(f"\n[INADMISSIBLE TRANSFORMS]")
        print(f"Failing rules: {a.failing_rules}")
        if a.suggestions:
            print(f"\nSuggestions:")
            for s in a.suggestions[:3]:
                print(f"  - {s}")

    # Show metrics
    if result.discovery:
        metrics = result.discovery.metrics
    elif result.falsification:
        metrics = result.falsification.metrics
    elif result.underdetermined:
        metrics = result.underdetermined.metrics
    else:
        metrics = None

    if metrics and metrics.metrics:
        print(f"\n{'='*60}")
        print("METRICS:")
        for m in metrics.metrics:
            ci = f" (CI95: {m.ci95[0]:.2f}-{m.ci95[1]:.2f})" if m.ci95 else ""
            print(f"  {m.name.value}: {m.value:.3f}{ci}")

    return result


# Preset experiments
EXPERIMENTS = {
    "1": {
        "name": "Concept: Justice",
        "claim": "The concept 'justice' has stable semantic structure",
        "positives": [
            "Justice requires treating equals equally",
            "Fairness is the foundation of justice",
            "Justice means giving each their due",
            "A just society protects the vulnerable",
            "Justice balances rights and responsibilities",
        ],
        "type": HypothesisType.CONCEPT_INVARIANT,
    },
    "2": {
        "name": "Relation: Cause-Effect",
        "claim": "Causal relationships preserve directionality under paraphrase",
        "positives": [
            "Smoking causes cancer",
            "The rain made the ground wet",
            "Exercise leads to better health",
            "Poverty results in poor education",
            "Heat causes ice to melt",
        ],
        "type": HypothesisType.RELATION_LAW,
    },
    "3": {
        "name": "Weak claim (should falsify)",
        "claim": "All sentences have identical meaning",
        "positives": [
            "The cat sat on the mat",
            "Quantum mechanics is hard",
            "I love pizza",
            "The stock market crashed",
            "She ran to the store",
        ],
        "type": HypothesisType.CONCEPT_INVARIANT,
    },
    "4": {
        "name": "Cluster: Emotions",
        "claim": "Positive emotions cluster together in semantic space",
        "positives": [
            "I feel so happy today",
            "This brings me great joy",
            "I'm excited about the future",
            "Love fills my heart",
            "Gratitude overwhelms me",
        ],
        "type": HypothesisType.CLUSTER_STRUCTURE,
    },
    "5": {
        "name": "Custom claim",
        "claim": None,  # Will prompt
        "positives": None,
        "type": HypothesisType.CONCEPT_INVARIANT,
    },
}


if __name__ == "__main__":
    print("\n" + "="*60)
    print("INVARIANT ENGINE - Interactive Demo")
    print("="*60)
    print("\nUsing STUB providers (no API calls)")
    print("Real providers: swap LLMProvider + EmbeddingProvider\n")

    print("Choose an experiment:\n")
    for k, v in EXPERIMENTS.items():
        print(f"  [{k}] {v['name']}")

    print("\n  [q] Quit\n")
