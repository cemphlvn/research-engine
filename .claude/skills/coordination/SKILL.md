---
name: coordination
description: "Manage multi-agent work. Use when delegating to multiple agents, handling handoffs, or resolving conflicts between approaches."
---

# Coordination

Orchestrate the orchestra.

## When to Coordinate

- Multiple agents involved
- Work handoffs needed
- Parallel work streams
- Conflicting approaches
- Integration points

## Coordination Patterns

### Sequential Handoff
```
Agent A completes → passes context → Agent B continues

Example:
meta-agent scaffolds → api-agent implements → test-agent validates
```

### Parallel Merge
```
Agent A works on X ─┐
                    ├─→ Orchestrator merges
Agent B works on Y ─┘

Example:
ui-agent builds frontend ─┐
                          ├─→ integrate
api-agent builds backend ─┘
```

### Review Loop
```
Agent A produces → Orchestrator critiques → Agent A revises

Example:
api-agent implements → criticism skill reviews → api-agent fixes
```

## Handoff Protocol

When delegating:
1. **Context**: What does the agent need to know?
2. **Scope**: What exactly should they do?
3. **Constraints**: What should they NOT do?
4. **Output**: What should they produce?

```markdown
## Delegation: [AGENT]

### Context
[Relevant background]

### Task
[Specific ask]

### Constraints
- Don't modify X
- Stay within scope Y

### Expected Output
[What success looks like]
```

## Conflict Resolution

When agents disagree or produce incompatible work:
1. Understand both approaches
2. Evaluate against project context
3. Decide (or escalate to user)
4. Document decision in state.jsonl

## Integration Checkpoints

Before merging work:
- Does it fit together?
- Any conflicts?
- Missing pieces?
- Ready for next phase?

## Log It

```jsonl
{"ts":"..","type":"delegate","msg":"delegated [task] to [agent]"}
{"ts":"..","type":"decision","msg":"resolved conflict: chose [approach] because [reason]"}
```
