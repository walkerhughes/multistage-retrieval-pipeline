# /reflect - Session Learning Capture

Analyze this conversation to identify key learnings about technologies, patterns, and preferences that should be captured as project-specific skills.

## Your Task

1. **Analyze the conversation** for moments where:
   - The user corrected your approach or rejected a suggestion
   - The user expressed frustration with a default behavior
   - A specific pattern or convention was established for this repo
   - A technique was approved that differs from common/default approaches
   - Tips, tricks, or gotchas were discovered during implementation

2. **Identify technologies** involved (e.g., FastAPI, PostgreSQL, pytest, Docker, etc.)

3. **Check existing skills** in `.claude/skills/` to see what already exists

4. **Determine what to create or update**:
   - **CREATE** a new skill if: A key learning exists about a technology where NO skill file exists yet, AND the learning represents repo-specific knowledge not in Claude's default training
   - **UPDATE** an existing skill if: New information was learned about a technology that already has a skill file

5. **Propose changes** to the user for approval (unless `--auto` flag was passed)

## Skill File Format

When creating or updating skills, use this structure:

**Location**: `.claude/skills/<technology>/SKILL.md`

```yaml
---
name: <technology>
description: Repo-specific conventions and patterns for <technology> in this codebase. Use when working with <technology> in this project.
---

# <Technology>

## Conventions
<!-- Naming conventions, file organization, structural patterns -->

## Approved Patterns
<!-- Patterns and approaches the user has accepted/preferred -->

## Anti-patterns
<!-- Approaches to avoid, things the user has rejected or corrected -->

## Tips & Tricks
<!-- Useful techniques, gotchas, and lessons learned -->
```

## Output Format

Present your findings like this:

### Session Analysis

**Technologies touched**: [list]

**Key learnings identified**:
1. [Learning 1 - which technology, what was learned]
2. [Learning 2 - ...]

### Proposed Skill Changes

For each change, show:

**[CREATE/UPDATE] `.claude/skills/<technology>/SKILL.md`**

```markdown
[Full content of the skill file or the specific sections to add]
```

Then ask: "Would you like me to apply these changes? (yes/no/edit)"

## Flags

- `--auto`: Skip approval and automatically apply all proposed changes
- Default (no flag): Show proposed changes and wait for user approval

## Important Guidelines

- Only capture learnings that are SPECIFIC to this repo, not general best practices
- Focus on USER PREFERENCES and CORRECTIONS, not just what was implemented
- Keep skill content concise and actionable
- One skill file per technology
- If no meaningful learnings exist, say so and don't create empty skills

## Example Learnings to Capture

- "User prefers pytest fixtures over setup/teardown methods"
- "API endpoints should return timing information in responses"
- "Always use connection pooling, never create connections per-request"
- "User got frustrated when I added docstrings to every function - only add them when logic isn't self-evident"
- "This repo uses `make` commands for common operations, not raw CLI commands"

