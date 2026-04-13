#!/usr/bin/env bash
# run_gate1.sh — Sequential Gate 1 harness (T01–T05 + verdict)
set -euo pipefail
cd "$(dirname "$0")"

PLAN="PLAN-1a-11.md"
PROMPTS_DIR="phase1_artifacts/prompts"
mkdir -p "$PROMPTS_DIR"

# ── Extract prompts ──────────────────────────────────────────────────────────

extract_prompt() {
    local n="$1"
    local out="$PROMPTS_DIR/t${n}_prompt.txt"
    awk "
        /^## Task ${n} Prompt/{found=1; next}
        found && /^\`\`\`$/ && !inside {inside=1; next}
        found && /^\`\`\`$/ && inside {exit}
        inside {print}
    " "$PLAN" > "$out"
    local lines
    lines=$(wc -l < "$out" | tr -d ' ')
    echo "T0${n} prompt: ${lines} lines → ${out}"
    if [ "$lines" -eq 0 ]; then
        echo "ERROR: Empty prompt for T0${n}. Check PLAN-1a-11.md format."
        exit 1
    fi
}

for i in 1 2 3 4 5 6; do
    extract_prompt "$i"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  All prompts extracted. Running T01–T05 sequentially..."
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# ── Run each test ────────────────────────────────────────────────────────────

for i in 1 2 3 4 5; do
    echo ""
    echo "───────────────────────────────────────────────────────────────"
    echo "  STARTING T0${i}"
    echo "───────────────────────────────────────────────────────────────"
    echo ""
    claude --print "Read the file phase1_artifacts/prompts/t${i}_prompt.txt and follow those instructions exactly. Write the script, run it, report the verdict." 2>&1 | tee "phase1_artifacts/t0${i}_log.txt"
    echo ""
    echo "  T0${i} DONE"
    echo ""
done

# ── Verdict synthesis ────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  All tests complete. Running verdict synthesis..."
echo "═══════════════════════════════════════════════════════════════════"
echo ""

claude --print "Read the file phase1_artifacts/prompts/t6_prompt.txt and follow those instructions exactly. Write the script, run it, report the verdict." 2>&1 | tee phase1_artifacts/verdict_log.txt

# ── Final result ─────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  GATE 1 COMPLETE"
echo "═══════════════════════════════════════════════════════════════════"

if [ -f "phase1_artifacts/1a_11_verdict.json" ]; then
    python3 -c "
import json
v = json.load(open('phase1_artifacts/1a_11_verdict.json'))
print()
print('CRITICAL PATH VERDICT:', v['critical_path_verdict'])
print()
print(v['summary'])
print()
for step in v.get('next_steps', []):
    print('  ->', step)
"
else
    echo "WARNING: Verdict JSON not found. Check verdict_log.txt"
fi
