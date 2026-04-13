# Hyperion Investigation Tracker

This directory is a structured research notebook for the Hyperion validation. It uses Obsidian-compatible `[[wikilinks]]` so opening this folder in Obsidian gives you the graph view for free.

## Structure

```
investigation/
├── README.md          ← you are here
├── STATUS.md          ← living dashboard: what's alive, dead, and next
├── claims/            ← things the paper or we assert to be true
├── hypotheses/        ← competing explanations for observed phenomena
├── tests/             ← designed experiments (planned, running, or complete)
├── evidence/          ← empirical findings (numbers, not interpretations)
├── decisions/         ← what was decided and why
```

## How It Works

Every file is one atomic unit — one claim, one hypothesis, one test, one finding. Files link to each other with `[[wikilinks]]`. The STATUS.md file is the only file you need to read to know where things stand.

### File Naming

- `claims/C01-sae-mc-higher.md` — claims are numbered sequentially
- `hypotheses/H01-factor-exposure.md` — competing explanations
- `tests/T01-year-demeaned-spearman.md` — designed experiments
- `evidence/E01-spearman-rho-022.md` — empirical findings
- `decisions/D01-mc-not-right-metric.md` — resolved decision points

### Frontmatter

Every file has YAML frontmatter:

```yaml
---
id: C01
status: alive | dead | qualified | untested
layer: 1 | 2 | 3
depends_on: [C01, E03]
killed_by: E05       # if dead
tested_by: [T01, T02]
---
```

### Progressive Elimination

The core discipline every hypothesis starts alive. Tests are designed to KILL hypotheses, not confirm them. When evidence kills a hypothesis, mark it dead and record what killed it. When all but one hypothesis survives in a given space, that's your conclusion — but only for that space.

### The Three Layers

- **Layer 1:** Does the paper's claim hold? (replication fidelity)
- **Layer 2:** Do SAE features carry signal? (foundation question)
- **Layer 3:** Can SAE power analog retrieval for DD? (product question)

Layer 1 findings don't automatically propagate to Layer 3. Each layer has its own claims, hypotheses, and tests.

## Convention for Claude Sessions

Any Claude session working on Hyperion should:
1. Read `STATUS.md` first
2. Classify new findings/questions by layer before reacting
3. When a test produces results, update the relevant files AND STATUS.md
4. Never argue when you can measure — design a test instead
