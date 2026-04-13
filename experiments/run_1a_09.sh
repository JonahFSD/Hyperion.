#!/bin/bash
# Runner for 1a_09 cluster size control diagnostic
# Usage: bash run_1a_09.sh
# Or pipe to Claude Code: echo "$(cat .prompts/09.txt)" | claude --dangerously-skip-permissions

set -e
cd ~/hyperion
source hyperion-env/bin/activate
python 1a_09_cluster_size_control.py 2>&1 | tee phase1_artifacts/log_09.txt
echo ""
echo "=== Done. Results in phase1_artifacts/1a_cluster_size_control.json ==="
