"""CLI entry point for Invariant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from invariant import __version__
from invariant.core.orchestrator import Orchestrator, OrchestratorConfig
from invariant.models.schemas import EvaluationOutcome, HypothesisType

app = typer.Typer(
    name="invariant",
    help="Epistemological engine: test hypothesis survival under admissible transformations",
)
console = Console()


@app.command()
def version():
    """Show version."""
    console.print(f"invariant {__version__}")


@app.command()
def evaluate(
    claim: str = typer.Argument(..., help="The claim/hypothesis to test"),
    positives: list[str] = typer.Option(
        [],
        "--positive", "-p",
        help="Positive examples (repeat for multiple)",
    ),
    hypothesis_type: str = typer.Option(
        "concept",
        "--type", "-t",
        help="Type: concept, relation, cluster, causal, symmetry",
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON file"),
):
    """Evaluate a claim for invariance under transformations."""
    console.print(f"[bold]Evaluating:[/bold] {claim}")

    # Map type string to enum
    type_map = {
        "concept": HypothesisType.CONCEPT_INVARIANT,
        "relation": HypothesisType.RELATION_LAW,
        "cluster": HypothesisType.CLUSTER_STRUCTURE,
        "causal": HypothesisType.CAUSAL_RULE,
        "symmetry": HypothesisType.SYMMETRY,
    }
    hyp_type = type_map.get(hypothesis_type, HypothesisType.CONCEPT_INVARIANT)

    # Use claim as positive if none provided
    if not positives:
        positives = [claim]
        console.print("[dim]No positives provided, using claim as example[/dim]")

    console.print(f"[dim]Positives: {len(positives)} examples[/dim]")

    # Run evaluation
    config = OrchestratorConfig(seed=seed)
    orchestrator = Orchestrator(config)
    result = orchestrator.evaluate(
        claim=claim,
        positives=positives,
        hypothesis_type=hyp_type,
    )

    # Display result
    _display_result(result)

    # Save if requested
    if output:
        output.write_text(result.model_dump_json(indent=2))
        console.print(f"[green]Saved to {output}[/green]")


@app.command()
def demo():
    """Run interactive demo with preset experiments."""
    from invariant.core.orchestrator import Orchestrator, OrchestratorConfig
    from invariant.models.schemas import HypothesisType

    experiments = [
        {
            "name": "Concept: Justice",
            "claim": "The concept 'justice' has stable semantic structure",
            "positives": [
                "Justice requires treating equals equally",
                "Fairness is the foundation of justice",
                "Justice means giving each their due",
                "A just society protects the vulnerable",
            ],
            "type": HypothesisType.CONCEPT_INVARIANT,
        },
        {
            "name": "Relation: Cause-Effect",
            "claim": "Causal relationships preserve directionality",
            "positives": [
                "Smoking causes cancer",
                "Rain makes the ground wet",
                "Exercise leads to better health",
            ],
            "type": HypothesisType.RELATION_LAW,
        },
        {
            "name": "Weak claim (should falsify)",
            "claim": "All sentences have identical meaning",
            "positives": [
                "The cat sat on the mat",
                "Quantum mechanics is hard",
                "I love pizza",
            ],
            "type": HypothesisType.CONCEPT_INVARIANT,
        },
    ]

    console.print("\n[bold]INVARIANT ENGINE - Demo[/bold]\n")
    console.print("Choose experiment:\n")
    for i, exp in enumerate(experiments, 1):
        console.print(f"  [{i}] {exp['name']}")

    choice = typer.prompt("\nEnter number", type=int, default=1)
    if choice < 1 or choice > len(experiments):
        console.print("[red]Invalid choice[/red]")
        return

    exp = experiments[choice - 1]
    console.print(f"\n[bold]Running: {exp['name']}[/bold]")

    orchestrator = Orchestrator(OrchestratorConfig(seed=42))
    result = orchestrator.evaluate(
        claim=exp["claim"],
        positives=exp["positives"],
        hypothesis_type=exp["type"],
    )

    _display_result(result)


def _display_result(result):
    """Display evaluation result with styling."""
    outcome = result.outcome.value.upper()

    console.print(f"\n{'='*60}")
    if result.outcome == EvaluationOutcome.DISCOVERY:
        console.print(f"[bold green]OUTCOME: {outcome}[/bold green]")
    elif result.outcome == EvaluationOutcome.FALSIFIED:
        console.print(f"[bold red]OUTCOME: {outcome}[/bold red]")
    elif result.outcome == EvaluationOutcome.INADMISSIBLE:
        console.print(f"[bold magenta]OUTCOME: {outcome}[/bold magenta]")
    else:
        console.print(f"[bold yellow]OUTCOME: {outcome}[/bold yellow]")
    console.print("="*60)

    if result.outcome == EvaluationOutcome.DISCOVERY and result.discovery:
        d = result.discovery
        console.print(f"\n[green]Confidence: {d.confidence:.0%}[/green]")
        console.print(f"\n{d.invariant_formal}")
        console.print(f"\n{d.invariant_plain}")
        if d.boundary_conditions:
            console.print("\n[dim]Boundaries:[/dim]")
            for b in d.boundary_conditions[:3]:
                console.print(f"  - {b.description}")

    elif result.outcome == EvaluationOutcome.FALSIFIED and result.falsification:
        f = result.falsification
        console.print(f"\n[red]Failure: {f.failure_mode.value}[/red]")
        console.print(f"Explanation: {f.explanation}")
        if f.counterexample:
            console.print(f"Breaking transform: {f.counterexample.breaking_transform}")
        if f.suggested_refinements:
            console.print("\n[dim]Suggestions:[/dim]")
            for s in f.suggested_refinements[:3]:
                console.print(f"  - {s}")

    elif result.outcome == EvaluationOutcome.UNDERDETERMINED and result.underdetermined:
        u = result.underdetermined
        console.print(f"\n[yellow]Reason: {u.reason}[/yellow]")
        if u.suggested_experiments:
            console.print("\n[dim]Next steps:[/dim]")
            for e in u.suggested_experiments[:3]:
                console.print(f"  - {e}")

    elif result.outcome == EvaluationOutcome.INADMISSIBLE and result.admissibility:
        a = result.admissibility
        console.print(f"\n[magenta]Failing rules: {a.failing_rules}[/magenta]")
        if a.suggestions:
            for s in a.suggestions[:3]:
                console.print(f"  - {s}")

    # Show metrics
    metrics = None
    if result.discovery:
        metrics = result.discovery.metrics
    elif result.falsification:
        metrics = result.falsification.metrics
    elif result.underdetermined:
        metrics = result.underdetermined.metrics

    if metrics and metrics.metrics:
        console.print(f"\n{'='*60}")
        console.print("[bold]METRICS[/bold]")
        for m in metrics.metrics:
            ci = f" (CI: {m.ci95[0]:.2f}-{m.ci95[1]:.2f})" if m.ci95 else ""
            console.print(f"  {m.name.value}: {m.value:.3f}{ci}")


if __name__ == "__main__":
    app()
