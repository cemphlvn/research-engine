---
description: "Bootstrap project with meta-agentic principles. Run this first in any new project."
tools: Read, Write, Edit, Bash, AskUserQuestion
---

# Start

## Check Bootstrap State

Read `.claude/context.md`. If it contains `[PROJECT]` placeholders, this is first run.

## First Run

1. Ask user: "What are we building? (be specific: tech stack, purpose, constraints)"

2. Based on answer, update `.claude/context.md`:
   - Replace `[PROJECT]` with project name
   - Fill in domain, tech stack, goals

3. Analyze what agents/skills are needed for this domain

4. Create project-specific agents in `.claude/agents/`:
   - Use templates from `.claude/templates/`
   - Customize for project domain

5. Create domain skills in `.claude/skills/`:
   - Package domain knowledge
   - Reference relevant docs/patterns

6. Scaffold project structure:
   - Create appropriate directories (src/, lib/, etc.)
   - Create initial files (package.json, Cargo.toml, etc.)
   - Create README.md with project description

7. Log bootstrap:
```bash
echo '{"ts":"'$(date -u +%FT%TZ)'","type":"bootstrap","msg":"initialized [PROJECT] with [AGENTS] agents"}' >> .claude/state.jsonl
```

## Continuing Run

If already bootstrapped:

1. Load `.claude/context.md` for project awareness
2. Load recent `.claude/state.jsonl` for continuity
3. Announce ready state with available agents/commands
