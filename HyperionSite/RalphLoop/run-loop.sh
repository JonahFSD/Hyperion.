#!/bin/bash
# Ralph loop runner — autonomous AI task execution
# Usage: bash agents/reusable/run-loop.sh agents/landing-interactive-demo
# Requires: PLAN.md and prompt.md in the target folder
# After all tasks complete, runs a diff audit agent automatically.

# NOTE: intentionally no `set -e` — this script kills subprocesses,
# which produces non-zero exit codes that would terminate the loop.

# Require a folder argument
if [ -z "$1" ]; then
    echo "Usage: bash agents/reusable/run-loop.sh <prompt-folder>"
    echo "Example: bash agents/reusable/run-loop.sh agents/landing-interactive-demo"
    exit 1
fi

FOLDER="$1"
PLAN="$FOLDER/PLAN.md"
PROMPT="$FOLDER/prompt.md"

if [ ! -f "$PLAN" ]; then
    echo "Error: $PLAN not found"
    exit 1
fi
if [ ! -f "$PROMPT" ]; then
    echo "Error: $PROMPT not found"
    exit 1
fi

# Prevent macOS from sleeping
caffeinate -dims -w $$ &
CAFFEINATE_PID=$!
echo "Caffeinate active (PID $CAFFEINATE_PID) — Mac will not sleep"

# Block ALL git write operations
FAKE_GIT_DIR=$(mktemp -d)
cat > "$FAKE_GIT_DIR/git" << 'WRAPPER'
#!/bin/bash
case "$1" in
    status|diff|log|show|branch)
        /usr/bin/git "$@"
        ;;
    *)
        echo "git $1 blocked by run-loop.sh"
        exit 0
        ;;
esac
WRAPPER
chmod +x "$FAKE_GIT_DIR/git"

export PATH="$FAKE_GIT_DIR:$PATH"

echo "=== Ralph loop starting at $(date) ==="
echo "=== Folder: $FOLDER ==="

while true; do
    if ! grep -q '^\- \[ \]' "$PLAN" 2>/dev/null; then
        echo "All tasks complete. Stopping."
        break
    fi

    # Snapshot current checked count (tr -cd strips anything non-numeric)
    BEFORE=$(grep -c '^\- \[x\]' "$PLAN" 2>/dev/null | tr -cd '0-9')
    BEFORE=${BEFORE:-0}

    echo "--- Starting next task at $(date) ---"
    cat "$PROMPT" | claude --dangerously-skip-permissions &
    CLAUDE_PID=$!

    # Monitor PLAN.md — when a new task gets checked off, kill claude
    while true; do
        sleep 5
        if ! kill -0 $CLAUDE_PID 2>/dev/null; then
            break
        fi
        AFTER=$(grep -c '^\- \[x\]' "$PLAN" 2>/dev/null | tr -cd '0-9')
        AFTER=${AFTER:-0}
        if [ "$AFTER" -gt "$BEFORE" ]; then
            echo "--- Task checked off in PLAN.md, killing process ---"
            kill $CLAUDE_PID 2>/dev/null
            sleep 2
            kill -9 $CLAUDE_PID 2>/dev/null
            break
        fi
    done

    wait $CLAUDE_PID 2>/dev/null
    echo "--- Task exited at $(date) ---"
    sleep 3
done

# ── Phase 2: Diff Audit ──
AUDIT_PROMPT="agents/reusable/diff-audit.md"
AUDITS_DIR="agents/audits"
if [ -f "$AUDIT_PROMPT" ]; then
    mkdir -p "$AUDITS_DIR"

    # Build timestamped filename: 2026-03-02-landing-interactive-demo.md
    FOLDER_LABEL=$(basename "$FOLDER")
    AUDIT_DATE=$(date +%Y-%m-%d)
    AUDIT_FILE="$AUDITS_DIR/${AUDIT_DATE}-${FOLDER_LABEL}.md"

    echo ""
    echo "=== Diff audit starting at $(date) ==="
    echo "=== Auditing changes from: $FOLDER ==="
    echo "=== Report will be written to: $AUDIT_FILE ==="

    # Build the audit prompt with the task folder path and output path injected
    AUDIT_INPUT="Read the diff audit prompt at $AUDIT_PROMPT. The task folder to audit is: $FOLDER. Write the report to $AUDIT_FILE."

    echo "$AUDIT_INPUT" | claude --dangerously-skip-permissions &
    AUDIT_PID=$!

    # Give the audit agent up to 10 minutes, then kill it
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
        # Also stop if audit file appears (audit is done writing)
        if [ -f "$AUDIT_FILE" ]; then
            # Give it a few seconds to finish writing
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
else
    echo "No diff-audit.md found at $AUDIT_PROMPT — skipping audit phase."
fi

# Cleanup
rm -rf "$FAKE_GIT_DIR"
kill $CAFFEINATE_PID 2>/dev/null
echo "=== Loop complete at $(date) ==="
echo "All changes are local and uncommitted."
if [ -n "$AUDIT_FILE" ] && [ -f "$AUDIT_FILE" ]; then
    echo "Review the audit report: $AUDIT_FILE"
fi
