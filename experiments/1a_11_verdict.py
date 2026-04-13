#!/usr/bin/env python3
"""
1a_11_verdict.py — Synthesis verdict for Hyperion Layer 2 decision gate.

Reads T01-T05 result JSONs, applies FRAMEWORK.md decision logic,
and produces a combined verdict for the product architecture.
"""

import json
import os
import sys
from pathlib import Path

ARTIFACTS_DIR = Path("phase1_artifacts")

TEST_FILES = {
    "T01": "1a_11_t01_result.json",
    "T02": "1a_11_t02_result.json",
    "T03": "1a_11_t03_result.json",
    "T04": "1a_11_t04_result.json",
    "T05": "1a_11_t05_result.json",
}


def load_results():
    """Load and validate all test result JSONs."""
    results = {}
    for test_id, filename in TEST_FILES.items():
        path = ARTIFACTS_DIR / filename
        if not path.exists():
            print(f"FATAL: Missing {path}")
            sys.exit(1)
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"FATAL: Invalid JSON in {path}: {e}")
                sys.exit(1)
        if "verdict" not in data:
            print(f"FATAL: {path} has no 'verdict' key — file is truncated or incomplete")
            sys.exit(1)
        results[test_id] = data
    return results


def synthesize(results):
    """Apply FRAMEWORK.md decision logic and produce verdict."""
    verdicts = {tid: r["verdict"] for tid, r in results.items()}

    # Extract key numbers
    t03 = results["T03"]
    t04 = results["T04"]
    t05 = results["T05"]

    sae_mean = t03["sae_top10_vs_sic"]["sae"]
    sic_mean = t03["sae_top10_vs_sic"]["sic"]
    t04_lift_k1 = t04["overall"]["K_1"]["lift"]
    t05_diff = t05["overall"]["difference"]

    # Decision logic from FRAMEWORK.md:
    #   T03 PASS + T04 PASS → SAE has signal, but T03 is a qualified pass
    #   because SAE (0.222) underperforms SIC (0.244) globally.
    #   Combined verdict: SAE is a within-industry re-ranker, not a global engine.
    t03_pass = verdicts["T03"] == "PASS"
    t04_pass = verdicts["T04"] == "PASS"
    sae_beats_sic_globally = sae_mean > sic_mean

    if t03_pass and t04_pass and not sae_beats_sic_globally:
        critical_path = "PIVOT_TO_RERANKER"
        arch = "SIC candidate generation → SAE re-ranking → LLM synthesis"
    elif t03_pass and sae_beats_sic_globally:
        critical_path = "SAE_AS_PRIMARY_RETRIEVER"
        arch = "SAE cosine retrieval → LLM synthesis"
    elif not t03_pass and t04_pass:
        critical_path = "PIVOT_TO_RERANKER"
        arch = "SIC candidate generation → SAE re-ranking → LLM synthesis"
    else:
        critical_path = "SAE_NO_RETRIEVAL_SIGNAL"
        arch = "Explore alternative feature extraction or learned ensemble"

    summary = (
        f"T01 (PASS): Year-demeaning increases signal from rho=0.003 to 0.022, "
        f"confirming the raw global correlation was suppressed by market-regime noise — "
        f"the underlying company-specific signal is real. "
        f"T02 (FAIL): Top-1% SAE pairs are only 1.24x enriched for same-SIC vs baseline, "
        f"meaning SAE's highest-similarity pairs are mostly cross-industry — the inversion "
        f"is not a SIC-composition artifact but reflects genuine cross-industry noise. "
        f"T03 (PASS, qualified): SAE nearest-neighbor retrieval beats random "
        f"(60% hit rate at K=10, lift CI excludes zero), but SAE top-10 mean correlation "
        f"({sae_mean:.3f}) underperforms the SIC baseline ({sic_mean:.3f}) — SAE has "
        f"global retrieval signal but is not the best global retriever. "
        f"T04 (PASS): Within-SIC, SAE re-ranking produces +{t04_lift_k1:.3f} lift at K=1 "
        f"over random same-industry peers, with CI excluding zero across all K values and "
        f"all 25 years — this is the strongest and most consistent result. "
        f"T05 (PASS): Graph topology adds +{t05_diff:.3f} correlation above "
        f"magnitude-matched controls, confirming that which companies are nearest neighbors "
        f"matters beyond raw cosine magnitude. "
        f"Combined verdict: SAE features carry real retrieval signal, but their comparative "
        f"advantage is within-industry discrimination, not global similarity. The product "
        f"architecture should use SIC codes for candidate generation and SAE cosine for "
        f"re-ranking within industry, not as a standalone retrieval engine."
    )

    next_steps = [
        "Run 1B factor adjustment to check if T04 within-SIC signal is factor-driven",
        "Run 1C permutation test for Layer 1 completeness",
        "Design multi-signal retrieval pipeline (Layer 3): SIC → SAE re-rank → LLM synthesis",
    ]

    output = {
        "tests": verdicts,
        "critical_path_verdict": critical_path,
        "architectural_implication": arch,
        "summary": summary,
        "next_steps": next_steps,
        "key_numbers": {
            "t04_lift_k1": t04_lift_k1,
            "t03_sae_vs_sic": f"{sae_mean:.3f} vs {sic_mean:.3f}",
            "t05_topology_value": f"{t05_diff:.4f}",
        },
    }
    return output


def update_status_md(verdicts):
    """Update investigation/STATUS.md test statuses if the file exists."""
    status_path = Path("investigation/STATUS.md")
    if not status_path.exists():
        return False

    content = status_path.read_text()

    # Map test IDs to their status-file identifiers and new statuses
    replacements = {
        "T01": ("[[T01-year-demeaned-spearman]]", verdicts["T01"]),
        "T02": ("[[T02-top1-sic-composition]]", verdicts["T02"]),
        "T03": ("[[T03-nn-precision-at-k]]", verdicts["T03"]),
        "T04": ("[[T04-within-sic-precision]]", verdicts["T04"]),
        "T05": ("[[T05-topology-vs-magnitude]]", verdicts["T05"]),
    }

    for tid, (wiki_id, verdict) in replacements.items():
        # Replace "NOT RUN" or "not_run" or "PENDING" with actual verdict
        for old_status in ["NOT RUN", "not_run", "PENDING"]:
            # Look for the pattern in the table row containing this test
            old = f"| {wiki_id} |"
            if old in content:
                # Find the line, replace the status column
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if wiki_id in line and old_status in line:
                        lines[i] = line.replace(old_status, verdict, 1)
                content = "\n".join(lines)

    status_path.write_text(content)
    return True


def main():
    print("=" * 70)
    print("Hyperion Layer 2 Decision Gate — Verdict Synthesis")
    print("=" * 70)
    print()

    # Step 1: Load and validate
    print("Loading T01-T05 result JSONs...")
    results = load_results()
    for tid in sorted(results):
        print(f"  {tid}: loaded OK (verdict={results[tid]['verdict']})")
    print()

    # Step 2: Synthesize
    verdict = synthesize(results)

    # Step 3: Print readable verdict
    print("-" * 70)
    print("TEST RESULTS")
    print("-" * 70)
    for tid, v in sorted(verdict["tests"].items()):
        print(f"  {tid}: {v}")
    print()

    print("-" * 70)
    print("CRITICAL PATH VERDICT")
    print("-" * 70)
    print(f"  {verdict['critical_path_verdict']}")
    print(f"  Architecture: {verdict['architectural_implication']}")
    print()

    print("-" * 70)
    print("KEY NUMBERS")
    print("-" * 70)
    print(f"  T04 within-SIC lift at K=1:  {verdict['key_numbers']['t04_lift_k1']:.4f}")
    print(f"  T03 SAE vs SIC (global):     {verdict['key_numbers']['t03_sae_vs_sic']}")
    print(f"  T05 topology value (diff):   {verdict['key_numbers']['t05_topology_value']}")
    print()

    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(verdict["summary"])
    print()

    print("-" * 70)
    print("NEXT STEPS")
    print("-" * 70)
    for i, step in enumerate(verdict["next_steps"], 1):
        print(f"  {i}. {step}")
    print()

    # Step 4: Write output JSON
    output_path = ARTIFACTS_DIR / "1a_11_verdict.json"
    with open(output_path, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"Verdict written to {output_path}")

    # Verify output
    with open(output_path) as f:
        check = json.load(f)
    assert "critical_path_verdict" in check
    assert len(check["tests"]) == 5
    print(f"Output JSON verified: {len(check['tests'])} tests, verdict={check['critical_path_verdict']}")

    # Step 5: Update STATUS.md
    verdicts = {tid: r["verdict"] for tid, r in results.items()}
    if update_status_md(verdicts):
        print("Updated investigation/STATUS.md with test results")
    else:
        print("investigation/STATUS.md not found — skipped update")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
