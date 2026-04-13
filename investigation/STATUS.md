# Status Board — 2026-03-26

## Hard Core Thesis

> Automated reading of SEC filings can produce a useful company-similarity signal at scale.

This is empirically obvious at the human level — doing DD on a company makes you smarter about what will happen to it. The question is whether you can automate the reading, quantify the similarity, and do it for thousands of companies. Everything below is about the protective belt — the specific methods used to operationalize this.

---

## Protective Belt (auxiliary hypotheses)

| ID | Hypothesis | Status | Evidence |
|----|-----------|--------|----------|
| [[H01-sae-captures-structure]] | SAE features capture structural similarity | UNTESTED at retrieval level | Global rho=0.022 (weak), within-SIC rho=0.117 (promising) |
| [[H02-cosine-is-right-metric]] | Cosine similarity is the right distance metric | UNTESTED | Top-1% inversion suggests nonlinear relationship |
| [[H03-mst-adds-value]] | MST+threshold adds value beyond raw similarity | UNTESTED | [[T05-topology-vs-magnitude]] designed |
| [[H04-mc-measures-quality]] | MC measures clustering quality fairly | DEAD | Killed by [[E04-pair-weighted-inversion]] |
| [[H05-signal-beyond-industry]] | SAE captures signal beyond industry membership | WEAK | Cross-SIC rho=0.016 ≈ zero |
| [[H06-signal-not-factors]] | SAE signal isn't just factor exposure | UNTESTED | Defers to 1B |

---

## Evidence Register

| ID | Finding | Numbers | Source |
|----|---------|---------|--------|
| [[E01-mc-replication]] | MC replicates paper exactly | SAE=0.359, SIC=0.231, SBERT=0.219 | 1a_02 |
| [[E02-delta-significant]] | SAE advantage statistically significant | All t > 5.5, p_BY < 0.001 | 1a_04 |
| [[E03-temporal-trend]] | SAE advantage growing over time | +0.006/yr, CI excludes zero | 1a_03 |
| [[E04-pair-weighted-inversion]] | SAE loses under pair-weighting | SAE 0.151 < SIC 0.252 < SBERT 0.210 | 1a_09 |
| [[E05-cluster-size-divergence]] | SAE clusters shrinking, SIC/SBERT growing | SAE 2.16→1.48, SIC 2.71→4.34, SBERT 14.1→29.1 | 1a_09 |
| [[E06-size-stratified-mc]] | SAE wins in small clusters, loses in large | Size 2: SAE 0.379 > SIC 0.240. Size 51+: SAE 0.206 < SIC 0.301 | 1a_09 |
| [[E07-global-spearman]] | Raw cosine has near-zero global predictive power | Spearman rho=0.022, 14.9M pairs | 1a_10 |
| [[E08-top1-inversion]] | Most similar pairs have below-baseline correlation | Top 1% corr=0.144 < baseline 0.161 | 1a_10 |
| [[E09-within-sic-signal]] | Signal exists within industries | Within-SIC rho=0.117 (mean), growing to 0.212 by 2020 | 1a_10 |
| [[E10-cross-sic-dead]] | No signal across industries | Cross-SIC rho=0.016 | 1a_10 |
| [[E11-sic-sbert-clones]] | SIC and SBERT are near-identical signals | Year-level r=0.954 | 1a_08 |
| [[E12-sae-different-signal]] | SAE captures something different from SIC/SBERT | SAE-SIC r=0.457, SAE-SBERT r=0.444 | 1a_08 |
| [[E13-ventile-hump]] | Cosine-correlation relationship is hump-shaped | Rises V5→V15, falls V15→V19, 8/19 violations | 1a_10 |

---

## Test Queue

### Priority 1 — Critical Path (Layer 2: do features carry signal?)

| ID | Test | Status | Resolves | Script |
|----|------|--------|----------|--------|
| [[T01-year-demeaned-spearman]] | Subtract year-mean correlation, recompute rho | PASS | Is rho=0.022 confounded by market regime? | 1a_11 |
| [[T02-top1-sic-composition]] | SIC codes of top-1% pairs vs base rate | FAIL | Is top-1% inversion from templates or real? | 1a_11 |
| [[T03-nn-precision-at-k]] | Nearest-neighbor return correlation vs random | PASS | Does retrieval work AT ALL? | 1a_11 |
| [[T04-within-sic-precision]] | Within-SIC NN correlation vs random peers | PASS | Does within-industry retrieval work? | 1a_11 |
| [[T05-topology-vs-magnitude]] | NN pairs vs magnitude-matched non-NN pairs | PASS | Does graph structure add value? | 1a_11 |

### Priority 2 — Layer 1 Completion (parallel)

| ID | Test | Status | Resolves | Script |
|----|------|--------|----------|--------|
| [[T06-month-alignment]] | Spot-check returns against Yahoo Finance | NOT RUN | HARD GATE for 1B | 1b_00 |
| [[T07-factor-adjustment]] | FF5 regression, residual MC | NOT RUN | Is signal explained by factors? | 1b_01-09 |
| [[T08-permutation-test]] | Feature-shuffle, rebuild MST, empirical p | NOT RUN | Is signal algorithmic artifact? | 1c |

### Priority 3 — Layer 3 (after Layer 2 resolves)

| ID | Test | Status | Resolves | Script |
|----|------|--------|----------|--------|
| [[T09-multi-signal-retrieval]] | SAE+SIC+SBERT+financials combined ranking | NOT DESIGNED | Does ensemble beat any single signal? | TBD |
| [[T10-factor-adjusted-retrieval]] | Repeat T03-T04 on factor-adjusted residuals | NOT DESIGNED | Does retrieval survive factor adjustment? | TBD |

---

## Decision Log

| ID | Decision | Date | Trigger | Consequence |
|----|----------|------|---------|-------------|
| [[D01-mc-not-right-metric]] | MC is not the right evaluation metric for Hyperion | 2026-03-26 | [[E04-pair-weighted-inversion]], [[E05-cluster-size-divergence]] | Pivot evaluation to precision@K retrieval metrics |
| [[D02-three-layer-framework]] | Separate paper-claim / feature-signal / product-utility evaluation | 2026-03-26 | Flags were conflating layers | Layer 2 is critical path, Layer 1 findings don't auto-propagate |
| [[D03-cross-sic-likely-dead]] | SAE probably can't find cross-industry analogs | 2026-03-26 | [[E10-cross-sic-dead]] | Hyperion retrieval is within-industry; cross-industry is a stretch goal, not core |

---

## Alive Hypothesis Space

These are the competing explanations that have NOT been killed. Every test should be designed to eliminate at least one.

### Why do SAE's tight clusters have high MC?

- **H-alive-1:** SAE genuinely identifies structurally similar companies within industries → would show up as positive lift in [[T04-within-sic-precision]]
- **H-alive-2:** SAE clusters group companies with similar factor loadings → would be killed by [[T07-factor-adjustment]]
- **H-alive-3:** Small clusters have high MC mechanically (fewer pairs, less regression to mean) → partially supported by [[E04-pair-weighted-inversion]], but [[E06-size-stratified-mc]] shows SAE beats SIC within same-size bins

### Why does the top 1% invert?

- **H-alive-4:** Extreme cosine similarity captures filing-template similarity, not business similarity → testable by [[T02-top1-sic-composition]]
- **H-alive-5:** The cosine-correlation relationship is genuinely nonlinear (hump-shaped) → supported by [[E13-ventile-hump]], would mean cosine is wrong metric at tails
- **H-alive-6:** Top-1% pairs are dominated by a few years with unusual market conditions → testable by year-stratified top-1% analysis (part of [[T01-year-demeaned-spearman]])

### What would make SAE useful for Hyperion?

- **H-alive-7:** SAE as sole retrieval signal (cosine → rank → return analogs) → tested by [[T03-nn-precision-at-k]]
- **H-alive-8:** SAE as within-industry discriminator (SIC for candidate gen, SAE for re-ranking) → tested by [[T04-within-sic-precision]]
- **H-alive-9:** SAE as one feature in a learned ensemble → requires [[T09-multi-signal-retrieval]], not yet designed
- **H-alive-10:** SAE graph topology as signal (MST structure, not raw cosine) → tested by [[T05-topology-vs-magnitude]]

---

## Execution Protocol

How ideas actually move from question to answer. This is the Ralph Loop adapted for research, not just code.

### The Cycle

```
OBSERVE → QUESTION → PREDICT → TEST → RECORD → UPDATE
```

1. **OBSERVE.** A number comes out of a script. Or a claim shows up in conversation. Write down the number. Do NOT interpret it yet.

2. **QUESTION.** What does this mean? What are the competing explanations? Write them ALL down, even the ones you don't like. If you can only think of one explanation, you haven't thought hard enough.

3. **PREDICT.** Before you write the test script, write down what each hypothesis predicts the result will be. "If H-alive-4 is correct (template similarity), then top-1% pairs should be mostly cross-SIC. If H-alive-5 is correct (nonlinear relationship), they should be mostly same-SIC but the within-SIC correlation should also be low." This is pre-registration. It prevents you from rationalizing after the fact.

4. **TEST.** One script. One JSON output. The script should be short enough that you can read every line. It should produce NUMBERS, not conclusions. The conclusion happens in your head after you read the numbers.

5. **RECORD.** Create an evidence file (E-next). Numbers only. What was measured, what came out. Separately note where your prediction was wrong — that's where the learning is.

6. **UPDATE.** Go through the alive hypothesis space. Is anything dead now? Is anything stronger? Update the status. If new questions emerged, add them. If a test needs to be designed, file it.

### Execution Heuristics

**Never interpret and observe in the same breath.** When a script finishes, first write down the numbers. Then close the terminal. Then think about what they mean. The gap prevents motivated reasoning.

**Every test should be designed to kill something.** If you can't name which hypothesis dies if the test comes back negative, the test isn't sharp enough. Refine it until it is.

**When you're confused, decompose.** If a result doesn't make sense, the question is too big. Break it into sub-questions that each have a clean test. "Does SAE work?" is not testable. "Does SAE cosine within-SIC-2digit-pharma predict return correlation at K=5?" is testable.

**Track your surprises explicitly.** Every time a result surprises you, write down: (a) what you expected, (b) what you got, (c) why you were wrong. The pattern of surprises tells you where your mental model is broken.

**Run the fastest test first.** If you have five tests designed, run the one that takes 30 seconds before the one that takes 3 hours. Fast feedback loops compound. The 30-second test might kill a hypothesis and make the 3-hour test unnecessary.

**Don't fall in love with a hypothesis.** If SAE features turn out to carry no signal, that's a FINDING, not a failure. The project pivots to a different feature extraction method. The hard core (automated filing reading → similarity at scale) is intact. Killing belt hypotheses quickly is how you get to the right architecture faster.

### Session Protocol

Any Claude session working on Hyperion:

1. Read this STATUS.md first — know what's alive, dead, and next
2. Classify new findings/questions by layer before reacting
3. Write predictions BEFORE running tests (in the test file's `predictions:` field)
4. When a test produces results: evidence file → hypothesis update → STATUS update
5. Never argue when you can measure — design a test instead

---

## What's Next

1. **Write and run 1a_11** (Tests T01-T05). This is the critical path. Everything else is parallel or blocked.
2. **Run 1B** (T06-T07) in parallel for Layer 1 completeness and to inform Layer 2.
3. **After Layer 2 resolves:** Design Layer 3 architecture based on which hypotheses survived.
