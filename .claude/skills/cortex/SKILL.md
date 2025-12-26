---
name: cortex
description: "Meta-agentic bootstrap system. Auto-loaded for all structural operations, agent creation, or when working with context.md/state.jsonl."
---

# Cortex

Self-building project bootstrap system.

## Axioms

1. **Structure = Ontology** - Filesystem IS knowledge graph
2. **Identity ≠ Location** - Agents know WHAT (agent.md) vs WHERE (context.md)
3. **Append-Only State** - state.jsonl tracks evolution across ALL sessions
4. **Bootstrap First** - System builds itself to build the project

## Required Files

| File | Purpose |
|------|---------|
| `context.md` | WHAT this project/folder IS |
| `state.jsonl` | Evolution log (persists across sessions) |

## state.jsonl Format

```jsonl
{"ts":"ISO","type":"bootstrap|created|modified|decision","msg":"..."}
```

## Bootstrap Flow

1. Clone cortex → new project
2. `/start` → meta-agent activates
3. "What are we building?" → user answers
4. meta-agent creates project-specific agents/skills
5. Those agents build the actual project

## Structure

```
project/
├── .claude/
│   ├── agents/
│   │   └── meta-agent.md      # permanent bootstrapper
│   │   └── [generated].md     # project-specific
│   ├── skills/
│   │   └── cortex/SKILL.md    # permanent principles
│   │   └── [generated]/       # project-specific
│   ├── commands/
│   ├── templates/
│   ├── context.md             # project ontology
│   └── state.jsonl            # project evolution
├── src/                       # actual code
└── README.md
```

## Before Any Write

1. Read `context.md` (location awareness)
2. Write
3. Log to `state.jsonl`
