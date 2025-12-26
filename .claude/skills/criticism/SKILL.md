---
name: criticism
description: "Evaluate work critically. Use after significant implementations, before finalizing decisions, or when something feels off."
---

# Criticism

The unexamined code is not worth shipping.

## When to Criticize

- After significant implementation
- Before major decisions
- When something feels wrong
- At integration points
- Before declaring "done"

## Criticism Dimensions

### Correctness
- Does it do what it should?
- Edge cases handled?
- Error cases handled?

### Completeness
- Is anything missing?
- Are all requirements met?
- What's been forgotten?

### Coherence
- Does it fit the architecture?
- Consistent with patterns?
- Makes sense with context.md?

### Simplicity
- Is this the simplest solution?
- Any unnecessary complexity?
- Could this be cleaner?

### Robustness
- What happens when things fail?
- How does it handle bad input?
- Is it resilient?

## Criticism Process

### 1. Step Back
Don't defend. Observe with fresh eyes.

### 2. Question Everything
- Why was this choice made?
- What alternatives exist?
- What assumptions are baked in?

### 3. Find Problems
Actively look for issues. Assume they exist.

### 4. Prioritize
- Critical: Must fix now
- Important: Should fix
- Minor: Could improve

### 5. Suggest
Don't just criticize. Propose solutions.

## Output Format

```markdown
## Critique: [WHAT]

### What's Good
- [Acknowledge what works]

### Issues

#### Critical
- Issue: [description]
  Fix: [suggestion]

#### Important
- Issue: [description]
  Fix: [suggestion]

#### Minor
- Issue: [description]

### Questions
- [Things to reconsider]

### Verdict
[Ship / Fix then ship / Rethink]
```

## Self-Criticism

The observer observes itself:
- Am I being thorough?
- Am I being fair?
- What am I missing?
- Are my criticisms actionable?

## Log It

```jsonl
{"ts":"..","type":"critique","msg":"critiqued [what]: [N] critical, [M] important issues"}
```
