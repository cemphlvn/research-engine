---
name: invariance-metrics
description: "Stability scoring, separability, compression metrics for invariance quantification"
---

# invariance-metrics

## Purpose

Higher-level metrics built on information theory to quantify invariance:
- Stability across transforms
- Separability under contrast probes
- Compression gain (MDL proxy)

## Patterns

### Multi-Transform Stability
```python
def multi_transform_stability(
    original: np.ndarray,
    transforms: list[Callable],
    model: EmbeddingModel
) -> StabilityReport:
    """Aggregate stability across multiple transforms."""
    scores = []
    for t in transforms:
        transformed = model.encode(t(original))
        scores.append(stability_score(original, transformed))
    return StabilityReport(
        mean=np.mean(scores),
        std=np.std(scores),
        min_transform=transforms[np.argmin(scores)],
        per_transform=dict(zip(transforms, scores))
    )
```

### Separability (Contrast Probing)
```python
def separability_score(
    positives: np.ndarray,
    negatives: np.ndarray,
    after_transform: bool = False
) -> float:
    """Can we distinguish true positives from adversarial negatives?"""
    # Train simple classifier (logistic reg)
    # Return AUC or accuracy
    # Key: if invariant is real, separability survives transform
```

### Compression Gain (MDL Proxy)
```python
def compression_gain(
    data: np.ndarray,
    invariant_basis: np.ndarray
) -> float:
    """How much does the invariant compress the data?"""
    # Project data onto invariant basis
    # Measure reconstruction error vs dimensionality reduction
    # Higher gain = more explanatory power
```

### Invariance Score Aggregation
```python
@dataclass
class InvarianceScore:
    stability: float      # Mean across transforms
    mi_preserved: float   # Information retention
    separability: float   # Contrast probe accuracy
    compression: float    # MDL proxy
    confidence: float     # Bootstrap CI width

    def overall(self, weights: dict = None) -> float:
        """Weighted combination of metrics."""
        w = weights or {"stability": 0.3, "mi": 0.3, "sep": 0.3, "comp": 0.1}
        return sum(getattr(self, k) * v for k, v in w.items())
```

## Examples

```python
# Full invariance evaluation
hypothesis = "The concept 'justice' has stable structure"
transforms = [paraphrase, add_context, synonym_swap]
positives = encode(justice_examples)
negatives = encode(adversarial_non_justice)

stability = multi_transform_stability(positives, transforms, model)
separability = separability_score(positives, negatives, after_transform=True)

if stability.mean > 0.8 and separability > 0.9:
    # Candidate for discovery
```

## References

- MDL: Rissanen, "Modeling by shortest data description"
- Contrast probing: Hewitt & Liang, "Designing and Interpreting Probes"
