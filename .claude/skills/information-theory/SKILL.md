---
name: information-theory
description: "MI, KL divergence, entropy calculations for invariance quantification"
---

# information-theory

## Purpose

Provide information-theoretic primitives for measuring invariance:
- Mutual Information: what structure is shared
- KL Divergence: how distributions drift
- Entropy: uncertainty/complexity measures

## Patterns

### Mutual Information
```python
def mutual_information(X: np.ndarray, Y: np.ndarray, bins: int = 30) -> float:
    """I(X;Y) = H(X) + H(Y) - H(X,Y)"""
    # For continuous: discretize or use KDE
    # For embeddings: use k-NN estimator (Kraskov)
```

### KL Divergence
```python
def kl_divergence(P: np.ndarray, Q: np.ndarray) -> float:
    """D_KL(P || Q) = sum(P * log(P/Q))"""
    # Handle zeros: add small epsilon or use smoothing
```

### Stability Score
```python
def stability_score(original: np.ndarray, transformed: np.ndarray) -> float:
    """Cosine similarity between representation distributions."""
    return cosine_similarity(original.mean(0), transformed.mean(0))
```

### Information Preserved Ratio
```python
def info_preserved(original: np.ndarray, transformed: np.ndarray) -> float:
    """I(f(X); f(T(X))) / H(f(X))"""
    mi = mutual_information(original, transformed)
    h = entropy(original)
    return mi / h if h > 0 else 0.0
```

## Examples

```python
# Measure if paraphrase preserves meaning
original_emb = model.encode(["The cat sat on the mat"])
para_emb = model.encode(["A feline rested on the rug"])
print(stability_score(original_emb, para_emb))  # Should be high

# Measure drift under context distractor
clean = model.encode(["Capital of France?"])
noisy = model.encode(["The weather is nice. Capital of France?"])
print(kl_divergence(clean, noisy))  # Should be low if robust
```

## References

- Kraskov MI estimator: doi:10.1103/PhysRevE.69.066138
- scipy.stats: entropy, gaussian_kde
- sklearn.metrics: mutual_info_score
