---
name: planning
description: "Decompose complex tasks into actionable plans. Use before any multi-step work, architectural decisions, or when the path forward is unclear."
---

# Planning

Think before acting. Decompose before executing.

## When to Plan

- Task has multiple steps
- Multiple approaches possible
- Dependencies between parts
- Uncertainty about path
- Significant effort required

## Planning Process

### 1. Understand
- What is actually being asked?
- What are the constraints?
- What does success look like?

### 2. Decompose
Break into atomic tasks:
```
GOAL: Build auth system

TASKS:
1. Design auth flow
2. Set up database schema
3. Implement login endpoint
4. Implement token management
5. Add middleware
6. Test
```

### 3. Sequence
Order by dependencies:
```
[1] → [2] → [3,4] → [5] → [6]
         ↘     ↗
          parallel
```

### 4. Identify Risks
- What could go wrong?
- What am I assuming?
- What don't I know?

### 5. Assign
Who does what:
- Orchestrator: planning, coordination
- meta-agent: structure, scaffolding
- domain-agent: domain work

## Output Format

```markdown
## Plan: [GOAL]

### Understanding
[What we're doing and why]

### Tasks
1. [ ] Task A (owner: X)
2. [ ] Task B (owner: Y, depends: A)
...

### Risks
- Risk 1: mitigation
- Risk 2: mitigation

### Open Questions
- Question needing answer before proceeding
```

## Log It

```jsonl
{"ts":"..","type":"plan","msg":"planned [GOAL]: N tasks, assigned to [agents]"}
```
