---
tags:
  - workflow
  - automation
  - ralph-loop
  - context-management
  - prompt-engineering
  - harness
  - back-pressure
status: current
---

# Workflow Architecture — Autonomous AI-Driven Development

Internal documentation for the CallSheet3 development workflow. This describes the systems, patterns, and principles used to run autonomous AI coding agents on this project.

---

## Core Concept: The Context Window Is an Array

Everything in this workflow derives from one insight: the LLM's context window is a fixed-size array. The less you put in it, the less the window needs to slide, and the better outcomes you get. Every decision below — fresh processes, minimal prompts, filesystem state, separated concerns — optimizes for keeping that array small and focused.

This is fundamentally different from the default approach of pounding the model in a loop until it hits compaction. Compaction is a lossy function. When the context window fills up and the system summarizes to make room, it can lose the "pin" — the critical frame of reference the model needs to do good work. Our approach avoids compaction entirely by killing the process and starting fresh.

---

## The Pin

The pin is the model's frame of reference for the current project. It's not injected into every context window — it exists on disk so the model can search it when needed.

**`agents/initialization/README.md`** is the primary pin. It's a lookup table organized by domain (Contacts, Billing, Calendar, Team, Auth, etc.) with keywords that map to file paths. The keywords aren't just labels — they include synonyms and related terms that improve the search tool's hit rate. "User authentication" also lists "login, JWT, Clerk, auth, session" so the model finds what it needs regardless of how the prompt phrases it.

**`CLAUDE.md`** is the conventions pin. Architecture principles, code style rules, file locations, DO/DON'T lists. This tells the model how to write code for this project, not what to write.

The pin doesn't get front-loaded into the context window. Individual task prompts carry their own file paths and context. The pin is there as a fallback when the model needs to search for something the prompt didn't specify. Loading the full pin into every iteration wastes array space.

---

## The Ralph Loop

Named after the pattern of running an AI agent in a deterministic loop that avoids context rot. The implementation lives in `agents/reusable/run-loop.sh`.

### How It Works

```
while true; do
    check PLAN.md for unchecked tasks → if none, stop
    spawn fresh claude process → reads driver prompt from stdin
    monitor PLAN.md → when checkbox changes, kill process
    sleep → loop restarts with fresh context
done
```

Each iteration is a completely independent process. No memory carries over except what's written to disk. The context window starts empty and ends when the process dies. There is no compaction because there is no accumulated context to compact.

### Why Kill Instead of Exit

Claude Code will not reliably exit after completing a task. Prompt-level instructions ("STOP", "exit", "do not continue") are ignored — the model wants to be helpful and starts the next task inside the same context window. This defeats the entire purpose of the loop.

The solution is external enforcement. The bash script runs Claude in the background, polls PLAN.md every 5 seconds, and kills the process the moment a new checkbox appears. The model doesn't need to cooperate. The harness decides when it's done.

### The Fake Git Wrapper

Claude Code will commit and push code regardless of prompt instructions telling it not to. "NEVER push" doesn't work. "Do NOT use git" doesn't work. The model has deeply ingrained patterns around git workflows that override explicit instructions.

The solution is a system-level intercept. `run-loop.sh` creates a temporary fake `git` binary that sits earlier in `$PATH` than the real one. It allows read-only commands (status, diff, log, show, branch) and silently blocks everything else (add, commit, push, checkout, reset). The model thinks its git commands succeeded. The harness ensures they didn't.

This is the single most important lesson from building this system: **back pressure belongs in the harness, not the prompt.** Asking the model to constrain its own behavior is unreliable. Blocking it at the system level is reliable.

### PLAN.md — The State Checkpoint

A markdown checklist that serves as shared memory between loop iterations. Format:

```markdown
## Status
- [x] 01-completed-task.md
- [x] 02-completed-task.md
- [ ] 03-next-task.md
- [ ] 04-future-task.md

## Blockers
Description of failures if verification didn't pass.
```

Each iteration reads it, finds the first unchecked item, does the work, and checks it off. The filesystem is the only state that persists. This is the "share memory by communicating, don't communicate by sharing memory" principle — iterations communicate through PLAN.md, not through shared context.

### The Driver Prompt

Lives at `agents/reusable/prompt.md`. Intentionally minimal — 7 lines. Earlier versions had extensive behavioral instructions that the model ignored. The final version just describes the workflow: find next task, read its prompt file, execute, verify, update PLAN.md.

The driver prompt should not include codebase context, architecture descriptions, or behavioral constraints. Task prompts carry their own context. The harness enforces behavioral constraints. The driver just routes.

---

## Specs-First Development

Features are not built by writing code. They're built through a conversation that generates specifications.

### The Pottery Wheel Metaphor

Building specs is like working clay on a pottery wheel. You start with a rough shape — "I want analytics like PostHog" — and incrementally refine through conversation. You test what the model knows, apply your engineering judgment, steer it away from bad assumptions, and shape the spec until it's precise enough to execute.

The conversation creates the spec. The human reviews and edits it. The spec goes into `specs/` with clear keywords for discoverability. Only then do you write prompts that reference the spec.

### Why Specs Before Prompts

The model needs a frame of reference to produce good code. Without a spec, it invents assumptions. With a spec, it follows a defined plan. The spec also makes the work reviewable — you can read a spec and understand what should be built without reading the code.

Specs don't need to be handcrafted from scratch. Generate them through conversation, then review and edit. The human's job is editorial and engineering judgment, not writing docs from a blank page.

### The Spec Lifecycle

1. Conversation generates a spec (planning session — one goal, one context window)
2. Human reviews and edits the spec
3. Spec gets indexed in `agents/initialization/README.md` (the pin)
4. Prompts reference the spec by path
5. Prompts get executed (execution session — separate context window)
6. Completed prompts move to `agents/archive/`

Planning and execution happen in different context windows. This is critical. The planning conversation accumulates context about requirements, tradeoffs, and design decisions. If you then execute in the same window, you've already consumed a large portion of the array on planning context that the execution doesn't need. Start fresh.

---

## Prompt Engineering

### The Prompt Structure

Every task prompt follows this format (defined in CLAUDE.md):

1. **Context** — What the problem is (2-3 paragraphs)
2. **Goal** — What success looks like (1-2 sentences)
3. **Files to Modify** — Explicit list, max 1-3 files
4. **Specific Changes** — Step-by-step with code examples
5. **DO / DO NOT** — Prevent scope creep
6. **Verify** — Grep + build commands to confirm the fix landed

### Why This Structure Works

Each section solves a specific failure mode. Context prevents the model from solving the wrong problem. Files to Modify prevents it from wandering into unrelated code. DO/DON'T prevents scope creep. Verify makes success measurable, not subjective.

The 8-80 line rule (per CLAUDE.md) isn't arbitrary. Below 8 lines, the overhead of a prompt isn't worth it. Above 80 lines, the model starts making mistakes because it's tracking too many changes at once. 1-3 files per prompt keeps the scope navigable.

### Discovery vs Execution Prompts

These are fundamentally different cognitive tasks that should never be mixed.

**Discovery prompts** (audits, analysis) read broadly across the codebase and produce reports. They're exploratory — the model searches, cross-references, and synthesizes findings. Output is a document, not code changes. Examples: `agents/reusable/tech-debt-audit.md`, `agents/reusable/security-audit.md`.

**Execution prompts** (fixes, features) read narrowly and produce code changes. They're precise — the model reads specific files and makes specific modifications. Output is changed source files. Examples: the 12 tech-debt prompts in `agents/archive/02-13-26-refactor/`.

Mixing them wastes time. The tech-debt refactor's prompt 01 spent 9 minutes re-auditing for dead code because the prompt told it to find and remove unused CSS. We already knew what was unused from the discovery audit. The prompt should have said "remove these specific classes" instead of "find what's unused." Discovery loop produces a report. Execution loop reads the report and acts on it.

---

## The Harness — External Enforcement

The most important architectural principle: **the harness controls behavior, not the prompt.**

### What the Harness Enforces

| Behavior | Prompt approach (unreliable) | Harness approach (reliable) |
|---|---|---|
| Don't push to remote | "NEVER run git push" | Fake git wrapper blocks push |
| Don't continue to next task | "STOP. Exit. You are done." | Kill process when PLAN.md changes |
| Don't stage prompt files | "Only stage source files" | Git is fully blocked, human stages |
| Don't run forever | "One task per run" | Process monitor + kill signal |
| Stop when all tasks done | "Exit if no unchecked tasks" | Bash checks PLAN.md before spawning |

Every time we tried to control behavior through the prompt, it failed. The model has strong default behaviors (push after commit, continue helping, stage everything) that override explicit instructions. The only reliable enforcement is system-level — fake binaries, process signals, conditional logic in the bash wrapper.

### Known Failure Modes

These were discovered through iteration and are solved by the current harness:

- **macOS doesn't have `timeout`** — caused an infinite error loop when the script tried to use it. Removed entirely.
- **Claude pushes despite being told not to** — fake git wrapper silently succeeds without pushing.
- **Claude continues to next task instead of exiting** — PLAN.md monitor kills the process after task completion.
- **Claude stages prompt files alongside source changes** — git write operations are fully blocked; human stages manually.
- **Claude tries to commit with co-author tags** — git commit is blocked at the system level.

---

## The Audit Agent — Post-Execution Verification

After the execution loop finishes (all PLAN.md tasks checked), a second agent spawns automatically in a fresh context window. This agent reads the `git diff`, checks it against project conventions and the task specs, and writes a structured report.

### Why a Separate Agent

The execution agent writes code. It's optimized for one task at a time, narrow scope, minimal context. Asking it to also verify its own work is a different cognitive task — it requires reading broadly across conventions, cross-referencing specs, and evaluating quality. These are discovery tasks, not execution tasks. Mixing them violates principle #6 (separate discovery from execution).

The audit agent also provides an independent check. The execution agent can't reliably evaluate whether it followed conventions it wasn't explicitly told about. The audit agent reads CLAUDE.md fresh and checks everything systematically.

### What It Checks

**Convention compliance** — every rule in CLAUDE.md, both backend (hex layering, account_id filtering, auth gates, input validation, error handling) and frontend (CSS variables, translations, naming conventions, style scoping, no new dependencies, dark mode).

**Spec compliance** — for each task prompt: was the goal met, were all steps implemented, were DO NOT constraints respected, do the verify commands pass.

**Scope** — did the execution agent stay within the declared "Files to Modify" list, or did it wander.

**Security** — new endpoints have auth, new input is validated, no secrets in source, tenant isolation enforced.

**General quality** — unused imports, console.log left in, dead code, obviously wrong patterns.

### How It Runs

Built into `run-loop.sh`. After the execution loop's `while true` exits (no more unchecked tasks), the script:

1. Checks for `agents/reusable/diff-audit.md`
2. Builds a timestamped output path: `agents/audits/YYYY-MM-DD-<task-label>.md`
3. Spawns a fresh Claude process with the audit prompt, task folder, and output path
4. Monitors for the audit file to appear (the report)
5. Kills the audit process after 10 minutes if it hasn't finished
6. Reports whether the audit produced a report

The audit agent runs under the same fake git wrapper — it can read diffs but can't commit or push. Its only output is the report file.

### The Report

Written to `agents/audits/YYYY-MM-DD-<task-label>.md` (e.g., `agents/audits/2026-03-02-landing-interactive-demo.md`). All audits accumulate in this folder as a persistent history. Structured format: summary (pass/warn/fail), convention violations with severity and line numbers, spec compliance per prompt, security review, and an ordered list of recommendations. The human reads this alongside the diff in GitHub Desktop before committing.

---

## Folder Structure

```
CallSheet3/
├── CLAUDE.md                          ← Immutable conventions (architecture, style, rules)
├── launch-claude.command              ← macOS double-click launcher for Claude Code CLI
├── agents/
│   ├── initialization/                ← Project knowledge & system docs
│   │   ├── README.md                 ← The pin — keyword-rich lookup table
│   │   ├── CLAUDE.md                 ← Conventions (architecture, style, rules)
│   │   ├── architecture.md           ← Full architecture documentation
│   │   └── workflow-architecture.md  ← This file
│   ├── reusable/                      ← Infrastructure that works across any batch
│   │   ├── run-loop.sh              ← The Ralph loop runner
│   │   ├── run-audit.sh             ← Standalone audit runner
│   │   ├── new-task.sh              ← Task folder scaffolding
│   │   ├── prompt.md                 ← Minimal driver prompt (legacy)
│   │   ├── tech-debt-audit.md        ← Quarterly codebase health check
│   │   ├── security-audit.md         ← Quarterly security posture check
│   │   └── diff-audit.md             ← Post-execution diff audit (runs automatically)
│   ├── audits/                        ← Timestamped audit reports (persistent history)
│   │   └── 2026-03-02-landing-interactive-demo.md
│   ├── archive/                       ← Completed prompts (DO NOT read during implementation)
│   │   └── 02-13-26-refactor/       ← 12 tech debt prompts, fully executed
│   └── specs/                         ← Additional specs (security audit reports, etc.)
```

### Why This Structure

- `agents/initialization/` is project knowledge. It's the reference material that any session can search.
- `agents/reusable/` is infrastructure. `run-loop.sh` and `prompt.md` work for any batch of tasks.
- `agents/archive/` is history. Completed prompts document what was done and why. The CLAUDE.md rule "DO NOT read during implementation" prevents context pollution.

---

## Immutable vs Mutable Configuration

### Immutable Layer (CLAUDE.md)

Things that are true regardless of what task is running. Architecture principles, code style rules, file locations, navigation patterns, CSS conventions, the feature checklist. This file only changes when project conventions change. Every Claude Code session reads it automatically.

### Mutable Layer (task-specific, manually updated)

Operational context that changes between sessions. Contains: what you're currently working on, what's been completed in the current effort, blockers or gotchas from the last session, and the immediate next step.

The human updates this manually before each session. This is a deliberate design decision — automated state updates (having Claude write its own context for the next session) introduce a telephone game where context degrades over iterations. The human maintains the source of truth.

Format is intentionally simple — 10-15 lines, four sections, overwritten frequently. Old state doesn't matter once you've moved past it.

---

## Running a Batch

### Setup

```bash
# Scaffold a new task folder with N prompt templates
bash agents/reusable/new-task.sh feature-xyz 4
```

This creates `agents/feature-xyz/` with PLAN.md, prompt.md (driver prompt with context.md support), and numbered template files. Then:

1. Rename the `XX-TODO.md` files to descriptive names (e.g., `01-add-endpoint.md`)
2. Update PLAN.md to match the new filenames
3. Fill in each prompt template with specific changes

If working from a discovery audit, read the audit report and translate each finding into a task prompt. The report says "what's wrong" — the prompt says "how to fix it, specifically." Don't tell the execution agent to "find and fix" — tell it to "change line X in file Y to Z." Discovery already found. Execution just acts.

### Execution

```bash
bash agents/reusable/run-loop.sh agents/feature-xyz
```

The loop runs until PLAN.md has no unchecked tasks, then automatically triggers the diff audit agent. `caffeinate` is built into the script. All changes are local and uncommitted.

### The context.md File

The driver prompt tells each iteration to read `context.md` (if it exists) and append notes for the next iteration. This is how discoveries survive process kills — if task 02 finds that a component was refactored differently than expected, it writes that to context.md so task 03 doesn't stumble on the same surprise.

context.md is append-only during a run. The execution agent adds notes; it never deletes them. After the batch completes, context.md gets archived with the rest of the folder.

### Post-Execution

1. Read the audit report at `agents/audits/YYYY-MM-DD-feature-xyz.md`
2. If the audit says **PASS** or **PASS WITH WARNINGS**: review diffs in GitHub Desktop, commit and push
3. If the audit says **NEEDS FIXES**: see "Handling Audit Failures" below
4. Move completed task folder to `agents/archive/`
5. Add a one-line summary to `agents/archive/INDEX.md`
6. Update `agents/initialization/README.md` if new files or domains were added

### Handling Audit Failures

When the audit flags issues that need fixing:

1. Read the audit report — each violation has a file, line number, and description
2. Create a new task folder for the fixes: `bash agents/reusable/new-task.sh feature-xyz-fixes N`
3. Write one prompt per fix (or group small related fixes into one prompt)
4. Run the loop: `bash agents/reusable/run-loop.sh agents/feature-xyz-fixes`
5. The audit runs again automatically on the new changes
6. Repeat until the audit passes

Do not mix fix prompts into the original task folder. The original folder is done — its PLAN.md is fully checked. Fixes are a new batch with their own PLAN.md, their own audit, and their own archive entry.

### Standalone Audit

To audit changes outside the loop (manual edits, one-off fixes):

```bash
bash agents/reusable/run-audit.sh agents/feature-xyz
```

This spawns the same audit agent with the same checks, writing to `agents/audits/`. If a report already exists for today, it auto-increments the filename (e.g., `-2.md`, `-3.md`).

### Discovery → Execution Handoff

Discovery audits (`tech-debt-audit.md`, `security-audit.md`) produce reports. Those reports need to become task prompts before any code changes happen.

The handoff process:

1. Run the discovery audit (e.g., `echo "Read agents/reusable/tech-debt-audit.md..." | claude --dangerously-skip-permissions`)
2. Read the report it produces
3. Group findings by domain or file — each group becomes one prompt
4. Scaffold the task folder: `bash agents/reusable/new-task.sh tech-debt-q2 N`
5. For each prompt, translate the finding into specific changes: file paths, line numbers, what to change and why
6. The prompt should reference the audit report for context but not ask the agent to re-discover the issue

The critical rule: **discovery says "what." Execution says "how."** If your execution prompt contains the word "find" or "search for," you're mixing concerns. The discovery audit already found it. Tell the execution agent exactly what to do.

---

## Economics

Each loop iteration on the Claude API costs pennies. Running 12 tech debt prompts overnight costs less than a coffee. On the Max plan ($200/month unlimited), it's effectively free.

A dedicated Mac Mini ($399-699 depending on RAM) running loops unattended is the optimal hardware setup. 16GB handles a single loop. 24GB gives headroom for future parallelization across branches.

The human investment is in prompt engineering, spec review, and diff review — not in writing code. The code is the cheap part now. The engineering judgment about what code to write, in what order, with what constraints — that's the expensive part that stays human.

---

## Principles Summary

1. **The context window is an array.** Minimize what goes in it.
2. **One goal per context window.** Don't mix planning and execution.
3. **Fresh context every iteration.** Kill the process. No compaction.
4. **The filesystem is the state.** PLAN.md is shared memory.
5. **Back pressure belongs in the harness.** System-level enforcement, not prompt instructions.
6. **Separate discovery from execution.** Audit first, then fix.
7. **The pin exists on disk, not in context.** Search it when needed, don't front-load it.
8. **Specs before prompts, prompts before code.** Shape the clay before firing it.
9. **8-80 lines per task, 1-3 files per prompt.** Stay in the reliable output range.
10. **The human is on the loop, not in the loop.** Engineering the back pressure is the job.
