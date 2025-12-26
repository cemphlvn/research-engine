---
name: transformation-design
description: "Build and validate transformation sets for invariance testing"
---

# transformation-design

## Purpose

Design transformation sets that are:
- Admissible (preserve evaluation + falsifiability)
- Diverse (cover multiple perturbation types)
- Graded (from mild to aggressive)

## Patterns

### Transform Families

| Family | Preserves | Destroys | Use When |
|--------|-----------|----------|----------|
| Paraphrase | Meaning | Surface form | Testing semantic invariance |
| Synonym | Denotation | Connotation sometimes | Testing lexical robustness |
| Context addition | Core claim | Focused attention | Testing distractor resistance |
| Reordering | Logical content | Sequential bias | Testing order independence |
| Negation | Form | Truth value | Contrast probe construction |
| Abstraction | Pattern | Specifics | Testing generalization |

### Admissibility Validation
```python
def validate_admissibility(transform_set: TransformSet) -> AdmissibilityResult:
    """
    Check:
    1. Evaluation preserved: apply to labeled examples, verify labels still meaningful
    2. Falsifiability preserved: apply to known-false, verify still detectable
    3. Information non-collapse: MI(original, transformed) > threshold
    """
    results = []
    for t in transform_set:
        eval_preserved = check_evaluation(t, labeled_examples)
        falsif_preserved = check_falsifiability(t, known_failures)
        info_preserved = check_information(t, test_corpus)
        results.append(TransformCheck(t, eval_preserved, falsif_preserved, info_preserved))

    return AdmissibilityResult(
        admissible=all(r.passes() for r in results),
        failing_transforms=[r.transform for r in results if not r.passes()],
        suggestions=generate_suggestions(results)
    )
```

### Graded Transform Chains
```python
# Mild → Aggressive ordering
GRADED_TRANSFORMS = [
    ("synonym_swap", 0.1),      # Very mild
    ("paraphrase", 0.3),        # Mild
    ("add_distractor", 0.5),    # Medium
    ("heavy_rewrite", 0.7),     # Strong
    ("adversarial_context", 0.9) # Aggressive
]

def find_breaking_point(hypothesis, transforms):
    """Binary search for the mildest transform that breaks invariance."""
    for name, severity in sorted(GRADED_TRANSFORMS, key=lambda x: x[1]):
        if not survives(hypothesis, name):
            return name, severity
    return None, 1.0  # Survives all
```

### Transform Composition
```python
def compose_transforms(*transforms) -> Transform:
    """Chain transforms: T1 ∘ T2 ∘ ... ∘ Tn"""
    def composed(x):
        for t in transforms:
            x = t(x)
        return x
    return Transform(
        name="+".join(t.name for t in transforms),
        apply=composed,
        expected_invariants=intersection(t.expected_invariants for t in transforms)
    )
```

## Examples

```python
# Build transform set for testing "concept stability"
concept_transforms = TransformSet([
    paraphrase_transform,
    synonym_swap_transform,
    context_distractor_transform,
])

# Validate before use
result = validate_admissibility(concept_transforms)
if not result.admissible:
    print(f"Remove: {result.failing_transforms}")
    print(f"Try: {result.suggestions}")
```

## References

- Ribeiro et al., "Beyond Accuracy: Behavioral Testing of NLP Models"
- Jia & Liang, "Adversarial Examples for Evaluating Reading Comprehension"
