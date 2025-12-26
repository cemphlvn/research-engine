# Invariant

## Domain

- **Type**: Research engine (CLI + library, web UI later)
- **Stack**: Python 3.11+, PyTorch, numpy, scipy, networkx, pydantic
- **Purpose**: Epistemological engine that tests whether hypotheses survive admissible transformations—outputs Discovery, Falsification, or Underdetermined

## Goals

1. Convert latent space "imagination" into testable knowledge via invariance constraints
2. Falsify unstable patterns with minimal counterexamples
3. Self-propel research by generating + testing hypotheses autonomously

## Constraints

- v0: text-only claims, paraphrase/context transforms
- Must be reproducible (seeded, logged runs)
- Human decides ethical admissibility; engine surfaces implications
- Fixed-point epistemology: admissible transforms ↔ surviving invariants must co-stabilize

## Core Concepts

| Concept | Definition |
|---------|------------|
| **Invariance** | Relational structure persisting under transformation class, measured as minimal info loss |
| **Admissible Transform** | Preserves evaluation (can still score) + falsifiability (can still fail) |
| **Fixed-Point** | Convergence when transforms and surviving invariants are mutually stable |

## Architecture

```
LLM Provider        = hypothesis gen + transform gen + adversary
Embedding Provider  = measurement instrument
Stats Engine        = epistemic court (hard gatekeeper)
Orchestrator        = scientific method in code (wires the above)
```

```
src/invariant/
├── providers/
│   ├── base.py          # LLMProvider, EmbeddingProvider abstractions
│   └── stats.py         # StatsEngine (epistemic court)
├── core/
│   ├── orchestrator.py  # Scientific method - wires providers together
│   ├── scoreboard.py    # ScoreboardSpec executor
│   ├── hypothesis.py    # Hypothesis utilities
│   ├── transforms.py    # Transformation library
│   ├── admissibility.py # Admissibility evaluator
│   ├── quantifier.py    # Legacy metrics (use StatsEngine)
│   ├── falsification.py # Counterexample generation
│   └── discovery.py     # Discovery synthesis
├── store/
│   └── experiment.py    # Run records, artifacts
├── models/
│   └── schemas.py       # Pydantic schemas (50+ types)
├── utils/
│   └── info_theory.py   # MI, KL, entropy primitives
└── cli.py               # Entry point
```

## Key Artifacts

| Artifact | Purpose |
|----------|---------|
| `ScoreboardSpec` | Executable evaluation DSL - the "rules of winning" |
| `Lens` | Representation through which invariance is measured (versioned) |
| `NegativeDataset` | First-class negatives (falsifiability requires teeth) |
| `ThresholdRule` | Structured decision rules (no string eval) |
| `DecisionPolicy` | falsify_if_any / discover_if_all / otherwise logic |
| `InvarianceScore` | Metrics with CI95 uncertainty |

## Agents

| Agent | Purpose |
|-------|---------|
| meta-agent | Bootstrap, structure, orchestration |
| engine-agent | Core pipeline: orchestrator, hypothesis gen, quantification |
| transforms-agent | Transformation library design + admissibility logic |
| falsification-agent | Counterexample generation, failure analysis |

## Skills

| Skill | Purpose |
|-------|---------|
| information-theory | MI, KL, entropy calculations |
| invariance-metrics | Stability scoring, separability, compression |
| transformation-design | Build/validate transform sets |

## Evolution

See `state.jsonl` for history.
