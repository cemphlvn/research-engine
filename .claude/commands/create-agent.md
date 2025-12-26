---
description: "Create a new project-specific agent"
tools: Read, Write, Edit
---

# Create Agent

## Usage

`/create-agent <name>` - creates `.claude/agents/<name>.md`

## Process

1. Read `.claude/templates/agent.md.tmpl`
2. Ask for:
   - Agent description (when to use)
   - Domain/scope
   - Key responsibilities
3. Generate `.claude/agents/<name>.md` with filled template
4. Log to `.claude/state.jsonl`

## Example

```
/create-agent api

? Description: Handles all REST API endpoints and validation
? Domain: backend
? Scope: src/api/

Creates: .claude/agents/api.md
```
