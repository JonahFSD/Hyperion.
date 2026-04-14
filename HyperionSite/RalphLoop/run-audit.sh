#!/bin/bash
# Run the diff audit agent standalone (outside the Ralph loop).
# Usage: bash agents/reusable/run-audit.sh <task-folder>
#
# Examples:
#   bash agents/reusable/run-audit.sh agents/landing-polish
#   bash agents/reusable/run-audit.sh agents/archive/landing-interactive-demo
#
# Useful for:
#   - Auditing manual changes before committing
#   - Re-auditing after fixing issues flagged by a previous audit
#   - Running against archived task folders retroactively

set -e

if [ -z "$1" ]; then
    echo "Usage: bash agents/reusable/run-audit.sh <task-folder>"
    echo "Example: bash agents/reusable/run-audit.sh agents/landing-polish"
    exit 1
fi

FOLDER="$1"
AUDIT_PROMPT="agents/reusable/diff-audit.md"
AUDITS_DIR="agents/audits"

if [ ! -f "$AUDIT_PROMPT" ]; then
    echo "Error: $AUDIT_PROMPT not found"
    exit 1
fi

if [ ! -d "$FOLDER" ]; then
    echo "Error: $FOLDER is not a directory"
    exit 1
fi

mkdir -p "$AUDITS_DIR"

# Build timestamped filename
FOLDER_LABEL=$(basename "$FOLDER")
AUDIT_DATE=$(date +%Y-%m-%d)
AUDIT_FILE="$AUDITS_DIR/${AUDIT_DATE}-${FOLDER_LABEL}.md"

# If a report already exists for today, add a suffix
if [ -f "$AUDIT_FILE" ]; then
    COUNTER=2
    while [ -f "$AUDITS_DIR/${AUDIT_DATE}-${FOLDER_LABEL}-${COUNTER}.md" ]; do
        COUNTER=$((COUNTER + 1))
    done
    AUDIT_FILE="$AUDITS_DIR/${AUDIT_DATE}-${FOLDER_LABEL}-${COUNTER}.md"
fi

echo "=== Diff audit starting at $(date) ==="
echo "=== Auditing changes from: $FOLDER ==="
echo "=== Report will be written to: $AUDIT_FILE ==="

AUDIT_INPUT="Read the diff audit prompt at $AUDIT_PROMPT. The task folder to audit is: $FOLDER. Write the report to $AUDIT_FILE."

echo "$AUDIT_INPUT" | claude --dangerously-skip-permissions &
AUDIT_PID=$!

# Give the audit agent up to 10 minutes
AUDIT_TIMEOUT=600
AUDIT_ELAPSED=0
while kill -0 $AUDIT_PID 2>/dev/null; do
    sleep 5
    AUDIT_ELAPSED=$((AUDIT_ELAPSED + 5))
    if [ "$AUDIT_ELAPSED" -ge "$AUDIT_TIMEOUT" ]; then
        echo "--- Audit timed out after ${AUDIT_TIMEOUT}s, killing ---"
        kill $AUDIT_PID 2>/dev/null
        sleep 2
        kill -9 $AUDIT_PID 2>/dev/null
        break
    fi
    if [ -f "$AUDIT_FILE" ]; then
        sleep 10
        kill $AUDIT_PID 2>/dev/null
        sleep 2
        kill -9 $AUDIT_PID 2>/dev/null
        break
    fi
done

wait $AUDIT_PID 2>/dev/null

if [ -f "$AUDIT_FILE" ]; then
    echo "=== Audit report written to $AUDIT_FILE ==="
else
    echo "=== Audit did not produce a report ==="
fi
