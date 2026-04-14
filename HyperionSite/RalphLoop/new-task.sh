#!/bin/bash
# Scaffold a new task folder for the Ralph loop.
# Usage: bash agents/reusable/new-task.sh <task-name> [number-of-prompts]
#
# Examples:
#   bash agents/reusable/new-task.sh landing-polish 4
#   bash agents/reusable/new-task.sh api-pagination          # defaults to 3 prompts
#
# Creates:
#   agents/<task-name>/
#   ├── PLAN.md           ← Checklist for the loop to monitor
#   ├── prompt.md         ← Driver prompt (piped to claude by run-loop.sh)
#   ├── 01-<task-name>.md ← Task prompt templates
#   ├── 02-<task-name>.md
#   └── ...

set -e

if [ -z "$1" ]; then
    echo "Usage: bash agents/reusable/new-task.sh <task-name> [number-of-prompts]"
    echo ""
    echo "  task-name          Kebab-case name (e.g. landing-polish, api-pagination)"
    echo "  number-of-prompts  How many task files to create (default: 3)"
    exit 1
fi

TASK_NAME="$1"
NUM_PROMPTS="${2:-3}"
FOLDER="agents/$TASK_NAME"

if [ -d "$FOLDER" ]; then
    echo "Error: $FOLDER already exists"
    exit 1
fi

mkdir -p "$FOLDER"

# ── Generate PLAN.md ──
{
    echo "## Status"
    for i in $(seq 1 "$NUM_PROMPTS"); do
        PADDED=$(printf "%02d" "$i")
        echo "- [ ] ${PADDED}-TODO.md"
    done
    echo ""
    echo "## Blockers"
    echo "None yet."
} > "$FOLDER/PLAN.md"

# ── Generate prompt.md (driver prompt for run-loop.sh) ──
cat > "$FOLDER/prompt.md" << EOF
Do NOT use git. No commits, no staging, no pushing. Only modify source files.

Read $FOLDER/PLAN.md. Find the first unchecked task.

If $FOLDER/context.md exists, read it for notes from previous iterations.

Read that prompt file from $FOLDER/. Execute the code changes it describes.

Run its verification steps.

If you discover anything the next iteration needs to know (unexpected file state, a dependency between tasks, a pattern you established), append it to $FOLDER/context.md.

Then mark that task as checked in PLAN.md.
EOF

# ── Generate numbered task prompt templates ──
for i in $(seq 1 "$NUM_PROMPTS"); do
    PADDED=$(printf "%02d" "$i")
    cat > "$FOLDER/${PADDED}-TODO.md" << 'TEMPLATE'
# Prompt PROMPT_NUM — [TITLE]

## Context

[What already exists. What this prompt builds on. Link to relevant files.]

## Goal

[One sentence. What does "done" look like?]

## Files to Modify

1. `path/to/file`

## Specific Changes

### 1. [Change description]

[Details. Code snippets if needed. Be precise — the execution agent reads this literally.]

## DO NOT

- Do not modify files outside the "Files to Modify" list
- Do not add new dependencies
- Do not change unrelated code

## Verify

```bash
# Build check
cd frontend && npx vite build 2>&1 | tail -5

# Manual check
# [describe what to look for in the browser]
```
TEMPLATE
    # Replace PROMPT_NUM placeholder with actual number
    sed -i "s/PROMPT_NUM/$PADDED/g" "$FOLDER/${PADDED}-TODO.md"
done

echo ""
echo "✓ Created $FOLDER/ with $NUM_PROMPTS task templates"
echo ""
echo "Next steps:"
echo "  1. Rename the task files:  mv $FOLDER/01-TODO.md $FOLDER/01-your-description.md"
echo "  2. Update PLAN.md to match the new filenames"
echo "  3. Fill in each prompt template"
echo "  4. Run:  bash agents/reusable/run-loop.sh $FOLDER"
echo ""
