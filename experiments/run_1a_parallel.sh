#!/bin/bash
# run_1a_parallel.sh — Phase 1A: write scripts in parallel, then run in wave order
#
# Phase A: 7 Claude Code instances write scripts simultaneously
# Phase B: Run Python scripts in dependency order (no Claude needed)
#
# Output: phase1_artifacts/1a_report.md, phase1_artifacts/1a_verdict.json
#
# Usage: bash run_1a_parallel.sh
# Requires: tmux, claude CLI, Python venv

set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"

SESSION="phase1a"
DONE_DIR="$REPO/phase1_artifacts/.done"
mkdir -p "$DONE_DIR"
rm -f "$DONE_DIR"/*.done

tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "═══════════════════════════════════════════════════════"
echo "  Phase 1A — Parallel Script Generation + Execution"
echo "═══════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════
# PHASE A: Write all 7 scripts in parallel
# ══════════════════════════════════════════════════

echo ""
echo "PHASE A: Writing 7 scripts in parallel..."
echo ""

# Each prompt is written to a temp file to avoid quoting hell in tmux send-keys
mkdir -p "$REPO/.prompts"

cat > "$REPO/.prompts/01.txt" << 'PROMPT'
You are writing ONE script for a 7-script pipeline. You are responsible for script 1 of 7.

YOUR TASK: Write the file 1a_01_data.py
DO NOT write any other 1a_*.py files. Do NOT write run_1a.sh. Only write 1a_01_data.py.

Read PLAN.md — specifically the section "### 1a_01_data.py — Data Loading".
Read company_similarity_sae/Clustering/GCD_Clustering_SAEs.py lines 60-109 for reference on MC computation.

The PLAN says this script "exists, verified, no changes needed." Check if the existing 1a_data.py matches the PLAN spec. If it does, copy it to 1a_01_data.py. If it doesn't, rewrite it to match.

Also write the report section file phase1_artifacts/1a_report_01.md per the PLAN (Observations then Interpretation sections).

Do NOT run the script. Just write the .py file and the report template.
PROMPT

cat > "$REPO/.prompts/02.txt" << 'PROMPT'
You are writing ONE script for a 7-script pipeline. You are responsible for script 2 of 7.

YOUR TASK: Write the file 1a_02_replicate.py
DO NOT write any other 1a_*.py files. Do NOT write run_1a.sh. Only write 1a_02_replicate.py.

Read PLAN.md — specifically the section "### 1a_02_replicate.py — MC Replication".
Read company_similarity_sae/Clustering/GCD_Clustering_SAEs.py lines 60-109 — this is the ground truth MC computation.
Read the existing 1a_replicate.py for reference.

Key change from existing script: you must compute BOTH mean AND median within-cluster correlation for all 7 methods. The output JSON schema is specified in the PLAN — follow it exactly.

Also write phase1_artifacts/1a_report_02.md per the PLAN (Observations then Interpretation).

Do NOT run the script. Just write the .py file and the report template.
PROMPT

cat > "$REPO/.prompts/03.txt" << 'PROMPT'
You are writing ONE script for a 7-script pipeline. You are responsible for script 3 of 7.

YOUR TASK: Write the file 1a_03_temporal.py
DO NOT write any other 1a_*.py files. Do NOT write run_1a.sh. Only write 1a_03_temporal.py.

Read PLAN.md — specifically the section "### 1a_03_temporal.py — Temporal Delta Analysis".

This script reads phase1_artifacts/1a_mc_by_year.json (which will exist when run but may not exist yet — that's fine, just write the code to read it).

It computes year-by-year deltas (SAE minus SIC, SAE minus SBERT), OLS regression of delta vs year, and bootstrap CI on the slope (10,000 resamples, seed 42). Output JSON schema is in the PLAN — follow it exactly.

Also write phase1_artifacts/1a_report_03.md per the PLAN (Observations then Interpretation).

Do NOT run the script. Just write the .py file and the report template.
PROMPT

cat > "$REPO/.prompts/04.txt" << 'PROMPT'
You are writing ONE script for a 7-script pipeline. You are responsible for script 4 of 7.

YOUR TASK: Write the file 1a_04_bootstrap.py
DO NOT write any other 1a_*.py files. Do NOT write run_1a.sh. Only write 1a_04_bootstrap.py.

Read PLAN.md — specifically the section "### 1a_04_bootstrap.py — Bootstrap CIs + Influence Diagnostics". Read it carefully — this is the most complex script.

This script:
1. Builds company-to-ticker mapping from companies.parquet
2. Precomputes flat arrays of within-cluster pairs for SAE, SIC, SBERT
3. VERIFICATION STEP: weighted MC with uniform weights must match 1a_replication.json within 1e-6. Halt if not.
4. Bootstrap: 10,000 iterations, resample tickers, BCa CIs (seed 42)
5. Delta tests: SAE-SIC, SAE-SBERT, SAE-baseline with p-values
6. Jackknife influence: leave-one-ticker-out, detect conclusion-flipping tickers
7. Output JSON schema is in the PLAN — follow it exactly

Also write phase1_artifacts/1a_report_04.md per the PLAN (Observations then Interpretation).

Do NOT run the script. Just write the .py file and the report template.
PROMPT

cat > "$REPO/.prompts/05.txt" << 'PROMPT'
You are writing ONE script for a 7-script pipeline. You are responsible for script 5 of 7.

YOUR TASK: Write the file 1a_05_theta.py
DO NOT write any other 1a_*.py files. Do NOT write run_1a.sh. Only write 1a_05_theta.py.

Read PLAN.md — specifically the section "### 1a_05_theta.py — Theta Sensitivity (Diagnostic Only)".

This is a DIAGNOSTIC script — not a formal test. It re-derives clusters from the cosine_similarity column in pairs.parquet to verify the shape of the theta curve.

Steps: compute cosine distance, StandardScaler on ALL years, build MSTs per year using scipy minimum_spanning_tree, sweep 100 thresholds (5th-95th percentile of MST edge weights), compute MC at each threshold, find optimal, also compute MC at ACL theta = -2.7.

Output JSON schema is in the PLAN — follow it exactly. No CV classification labels — just report the raw numbers.

Also write phase1_artifacts/1a_report_05.md per the PLAN (Observations then Interpretation).

Do NOT run the script. Just write the .py file and the report template.
PROMPT

cat > "$REPO/.prompts/06.txt" << 'PROMPT'
You are writing ONE script for a 7-script pipeline. You are responsible for script 6 of 7.

YOUR TASK: Write the file 1a_06_rolling.py
DO NOT write any other 1a_*.py files. Do NOT write run_1a.sh. Only write 1a_06_rolling.py.

Read PLAN.md — specifically the section "### 1a_06_rolling.py — Rolling Temporal Holdout".

This script reads phase1_artifacts/1a_mc_by_year.json and computes rolling 5-year windows: [1996-2000], [1997-2001], ..., [2016-2020] = 21 windows. For each window: mean SAE/SIC/SBERT MC and deltas. Reports win rates and worst-case windows. No arbitrary thresholds — just report numbers.

Output JSON schema is in the PLAN — follow it exactly.

Also write phase1_artifacts/1a_report_06.md per the PLAN (Observations then Interpretation).

Do NOT run the script. Just write the .py file and the report template.
PROMPT

cat > "$REPO/.prompts/07.txt" << 'PROMPT'
You are writing ONE script for a 7-script pipeline. You are responsible for script 7 of 7.

YOUR TASK: Write the file 1a_07_verdict.py
DO NOT write any other 1a_*.py files. Do NOT write run_1a.sh. Only write 1a_07_verdict.py.

Read PLAN.md — specifically the section "### 1a_07_verdict.py — BY FDR Correction + Report Assembly".

This script:
1. Reads ALL JSON files: 1a_replication.json, 1a_temporal.json, 1a_bootstrap.json, 1a_theta.json, 1a_rolling.json
2. Applies Benjamini-Yekutieli (BY) FDR correction on the 3 delta p-values ONLY (not BH — BY). Formula: threshold = (alpha * k) / (m * c_m) where c_m = sum(1/i for i in 1..m). Or use scipy.stats.false_discovery_control(ps, method='by') if scipy >= 1.11.
3. Computes HLZ t-statistics: t = delta / bootstrap_std
4. Evaluates 4 hard tests and 5 diagnostics per the PLAN table
5. Assembles 1a_report_01.md through 1a_report_06.md into phase1_artifacts/1a_report.md
6. Writes phase1_artifacts/1a_verdict.json matching the PLAN schema exactly

Do NOT run the script. Just write the .py file.
PROMPT

# Launch tmux with 7 panes
tmux new-session -d -s "$SESSION" -n "write"

for i in 1 2 3 4 5 6 7; do
  if [ "$i" -gt 1 ]; then
    tmux split-window -t "$SESSION:write"
    tmux select-layout -t "$SESSION:write" tiled
  fi
  PANE_IDX=$((i - 1))
  tmux send-keys -t "$SESSION:write.${PANE_IDX}" \
    "cd $REPO && echo '>>> Writing 1a_0${i}' && cat .prompts/0${i}.txt | claude --dangerously-skip-permissions -p && touch $DONE_DIR/write_0${i}.done && echo '>>> 1a_0${i} WRITTEN'" Enter
done

echo "  7 Claude instances launched in tmux session '$SESSION'"
echo "  Attach with: tmux attach -t $SESSION"
echo ""
echo "  Waiting for all scripts to be written..."

for i in 1 2 3 4 5 6 7; do
  while [ ! -f "$DONE_DIR/write_0${i}.done" ]; do sleep 2; done
  echo "    1a_0${i} written."
done

echo ""
echo "PHASE A complete."
echo ""

# ══════════════════════════════════════════════════
# Verify all 7 files exist before running
# ══════════════════════════════════════════════════

echo "Verifying all scripts exist..."
MISSING=0
for i in 1 2 3 4 5 6 7; do
  FILE="$REPO/1a_0${i}_"
  # Find the actual file (name varies per script)
  EXPECTED=(
    "$REPO/1a_01_data.py"
    "$REPO/1a_02_replicate.py"
    "$REPO/1a_03_temporal.py"
    "$REPO/1a_04_bootstrap.py"
    "$REPO/1a_05_theta.py"
    "$REPO/1a_06_rolling.py"
    "$REPO/1a_07_verdict.py"
  )
  if [ ! -f "${EXPECTED[$((i-1))]}" ]; then
    echo "  MISSING: ${EXPECTED[$((i-1))]}"
    MISSING=1
  else
    echo "  OK: ${EXPECTED[$((i-1))]}"
  fi
done

if [ "$MISSING" -eq 1 ]; then
  echo ""
  echo "ERROR: Some scripts were not written. Check tmux logs."
  exit 1
fi

echo ""

# ══════════════════════════════════════════════════
# PHASE B: Run Python scripts in wave order
# ══════════════════════════════════════════════════

echo "PHASE B: Running Python scripts..."
echo ""

source "$REPO/hyperion-env/bin/activate"

# --- Wave 1: Data (everything depends on this) ---
echo "  Wave 1/4: 01_data"
python "$REPO/1a_01_data.py" 2>&1 | tee "$REPO/phase1_artifacts/log_01.txt"
echo "    DONE"

# --- Wave 2: replicate + bootstrap + theta (parallel) ---
echo "  Wave 2/4: 02_replicate, 04_bootstrap, 05_theta (parallel)"

python "$REPO/1a_02_replicate.py" 2>&1 | tee "$REPO/phase1_artifacts/log_02.txt" &
PID_02=$!

python "$REPO/1a_04_bootstrap.py" 2>&1 | tee "$REPO/phase1_artifacts/log_04.txt" &
PID_04=$!

python "$REPO/1a_05_theta.py" 2>&1 | tee "$REPO/phase1_artifacts/log_05.txt" &
PID_05=$!

# Wait for 02 first (wave 3 needs 1a_mc_by_year.json)
wait $PID_02 || { echo "FAILED: 1a_02_replicate.py — see log_02.txt"; exit 1; }
echo "    02_replicate DONE"

# --- Wave 3: temporal + rolling (parallel, need 02's output) ---
echo "  Wave 3/4: 03_temporal, 06_rolling (parallel)"

python "$REPO/1a_03_temporal.py" 2>&1 | tee "$REPO/phase1_artifacts/log_03.txt" &
PID_03=$!

python "$REPO/1a_06_rolling.py" 2>&1 | tee "$REPO/phase1_artifacts/log_06.txt" &
PID_06=$!

# Wait for remaining wave 2 + wave 3
wait $PID_04 || { echo "FAILED: 1a_04_bootstrap.py — see log_04.txt"; exit 1; }
echo "    04_bootstrap DONE"

wait $PID_05 || { echo "FAILED: 1a_05_theta.py — see log_05.txt"; exit 1; }
echo "    05_theta DONE"

wait $PID_03 || { echo "FAILED: 1a_03_temporal.py — see log_03.txt"; exit 1; }
echo "    03_temporal DONE"

wait $PID_06 || { echo "FAILED: 1a_06_rolling.py — see log_06.txt"; exit 1; }
echo "    06_rolling DONE"

# --- Wave 4: Verdict (needs everything) ---
echo "  Wave 4/4: 07_verdict"
python "$REPO/1a_07_verdict.py" 2>&1 | tee "$REPO/phase1_artifacts/log_07.txt"
echo "    07_verdict DONE"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Phase 1A complete."
echo ""
echo "  Report:  phase1_artifacts/1a_report.md"
echo "  Verdict: phase1_artifacts/1a_verdict.json"
echo "  Logs:    phase1_artifacts/log_0N.txt"
echo ""
python3 -c "
import json
v = json.load(open('$REPO/phase1_artifacts/1a_verdict.json'))
print(f\"  Overall: {v['overall']}\")
print(f\"  Rationale: {v.get('rationale', 'N/A')}\")
"
echo "═══════════════════════════════════════════════════════"
