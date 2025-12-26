# Invariant Research Engine

A system that tests whether claims have real meaning by checking if they stay stable when you rephrase them.

## The Simple Explanation

**What problem does this solve?**

How do you know if a statement actually means something? Consider:
- "Water boils at 100°C at sea level" - clear, testable
- "Things happen in various ways sometimes" - sounds meaningful but says nothing
- "Colorless green ideas sleep furiously" - grammatically correct but nonsense

This system tests claims by asking: **does the meaning survive when you say it differently?**

**How it works:**

1. Take a claim like "Justice requires equal treatment"
2. Rephrase it several ways: "Equal treatment is required by justice", "For justice, all must be treated equally"
3. Generate counter-examples: "Justice permits unequal treatment in emergencies"
4. Measure: Do the rephrases stay close together? Are they clearly different from counter-examples?

If yes → the claim has stable meaning (DISCOVERY)
If no → the claim might be vague or incoherent (UNDERDETERMINED)

**Think of it like this:** A solid object keeps its shape when you rotate it. A meaningful claim keeps its meaning when you rephrase it.

---

## The Technical Explanation

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                         │
│  Coordinates the scientific method for semantic claims   │
└─────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │    LLM    │    │ EMBEDDING │    │   STATS   │
   │  Provider │    │  Provider │    │  Engine   │
   └───────────┘    └───────────┘    └───────────┘
        │                 │                 │
   Generates:        Measures:         Decides:
   - Transforms      - Similarity      - Stability
   - Negatives       - Distance        - Separability
   - Hypotheses      - Clustering      - KL Drift
```

### Core Concepts

**Invariance Testing**: A claim is meaningful if its semantic representation is invariant under meaning-preserving transformations.

**Metrics**:
| Metric | What it measures | Good value |
|--------|------------------|------------|
| Stability | Do rephrases stay close to original? | > 0.80 |
| Separability | Are negatives clearly different? | > 0.75 |
| KL Drift | Does transform distribution shift? | < 2.0 |
| Mutual Information | Is structure preserved? | > 0.50 |

**Outcomes**:
- `DISCOVERY` - Claim shows stable invariant structure
- `FALSIFIED` - Claim fails invariance tests
- `UNDERDETERMINED` - Metrics conflict, needs more data
- `INADMISSIBLE` - Transforms don't preserve meaning

### Providers

| Provider | Role | Default |
|----------|------|---------|
| DeepSeek | LLM for transforms/negatives | Primary |
| OpenAI | Embeddings (text-embedding-3-small) | Primary |
| Anthropic | Alternative LLM | Fallback |
| Stub | Testing without API | Fallback |

### Usage

```python
from invariant.core.orchestrator import Orchestrator

orch = Orchestrator()  # Auto-detects API keys

result = orch.evaluate(
    claim="Exercise improves mental health",
    positives=[
        "Exercise improves mental health",
        "Physical activity enhances psychological well-being",
        "Working out helps your mood",
    ]
)

print(result.outcome)  # → EvaluationOutcome.DISCOVERY
```

### Configuration

```bash
# .env
DEEPSEEK_API_KEY=sk-...      # For LLM (transforms, negatives)
OPENAI_API_KEY=sk-...        # For embeddings
ANTHROPIC_API_KEY=sk-ant-... # Optional alternative LLM
```

---

## Installation

```bash
git clone https://github.com/cemphlvn/research-engine.git
cd research-engine
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Quick Test

```bash
# With API keys
cp .env.example .env  # Add your keys
python demo.py

# Without API keys (uses stubs)
python -c "
from invariant.core.orchestrator import Orchestrator
orch = Orchestrator()
r = orch.evaluate('Water boils at 100C', ['Water boils at 100C', 'At 100C water boils'])
print(r.outcome)
"
```

---

## Contributing

### What we need help with

1. **Harder negatives** - Current negatives are too easy to distinguish (separability always ~1.0). Need true "near-miss" negatives that are semantically close but meaning-different.

2. **Coherence detection** - System can't tell nonsense from meaning. "Colorless green ideas sleep furiously" gets DISCOVERY because it's stable under transforms. Need explicit coherence metric.

3. **Better transforms** - Current transforms (paraphrase, formalize, simplify) may be too conservative. Explore more aggressive meaning-preserving transforms.

4. **Calibration** - Thresholds (kl_max=2.0, stability=0.80) are guesses. Need empirical calibration on labeled dataset.

### How to contribute

1. **Fork & clone**
   ```bash
   gh repo fork cemphlvn/research-engine --clone
   cd research-engine
   ```

2. **Set up dev environment**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Run tests**
   ```bash
   pytest tests/
   ```

4. **Make changes** - Focus on:
   - `src/invariant/providers/semantic.py` - Transform/negative generation
   - `src/invariant/providers/stats.py` - Metric computation
   - `src/invariant/core/orchestrator.py` - Pipeline logic

5. **Submit PR** with:
   - What problem you're solving
   - Before/after metrics if applicable
   - Tests for new functionality

### Code structure

```
src/invariant/
├── core/
│   └── orchestrator.py    # Main pipeline
├── providers/
│   ├── semantic.py        # LLM prompts for transforms/negatives
│   ├── stats.py           # Metrics (KL, stability, separability)
│   ├── openai_provider.py # OpenAI embeddings
│   └── base.py            # Abstract interfaces + stubs
├── models/
│   └── schemas.py         # Pydantic models
└── utils/
    └── info_theory.py     # KL divergence, mutual info
```

### Design principles

- **Stats layer is the gatekeeper** - LLMs generate, stats decide. Remove stats = storytelling engine.
- **Falsifiability over confirmation** - System should try to break claims, not confirm them.
- **Stub-first development** - Everything works without API keys for testing.

---

## License

MIT

## Acknowledgments

Built with Claude Code. Uses DeepSeek for reasoning, OpenAI for embeddings.
