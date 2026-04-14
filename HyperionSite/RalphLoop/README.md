# Agents — Autonomous AI Development System

This folder is the infrastructure for running autonomous AI coding agents on CallSheet3. If you're a new Claude session and Jonah asks you to make code changes, this is how it works.

---

## Why This Exists

The LLM context window is a fixed-size array. When it fills up, the system compacts (summarizes) to make room — and compaction is lossy. It drops details, forgets conventions, loses track of file paths. The longer a session runs, the worse the output gets.

This system avoids compaction entirely. Instead of one long session that degrades over time, we run many short sessions that each do one thing well. A shell script spawns a fresh Claude process for each task, kills it when the task is done, and spawns a new one for the next task. Every iteration starts with a clean context window. Nothing rots.

The tradeoff: each session can't remember what the last one did. So we use the filesystem as shared memory. PLAN.md tracks progress. context.md passes notes between iterations. The conventions live in files on disk, not in the prompt.

---

## How It's Organized

```
agents/
├── README.md                ← You are here
├── initialization/          ← Project knowledge (immutable during execution)
│   ├── CLAUDE.md           ← Code conventions, architecture rules, DO/DON'T
│   ├── README.md           ← The "pin" — keyword-rich file/route lookup table
│   ├── architecture.md     ← Full system architecture docs
│   └── workflow-architecture.md  ← Deep dive on this workflow system
├── reusable/                ← Scripts and audit prompts (shared across all batches)
│   ├── new-task.sh         ← Scaffold a new task folder
│   ├── run-loop.sh         ← The execution loop
│   ├── run-audit.sh        ← Standalone audit runner
│   ├── diff-audit.md       ← Post-execution audit prompt
│   ├── tech-debt-audit.md  ← Quarterly codebase health check
│   └── security-audit.md   ← Quarterly security posture check
├── audits/                  ← Timestamped audit reports (persistent history)
└── archive/                 ← Completed task folders + INDEX.md
```

**initialization/** contains docs that don't change during execution. CLAUDE.md tells Claude how to write code for this project (hex architecture, naming conventions, CSS variables, translations). README.md is a searchable lookup table mapping keywords to file paths. These exist on disk for the model to read when needed — they don't get front-loaded into every context window.

**reusable/** contains the scripts and prompts that work across any task batch. You never edit these for a specific task — you configure each batch through its own folder.

**audits/** accumulates reports over time. Every loop run produces a timestamped audit. This is persistent history — it doesn't get archived or cleaned up.

**archive/** is where completed task folders go when they're done. Each folder has an entry in archive/INDEX.md so you can find things later.

---

## The Two Types of Work

**Discovery** reads broadly and produces reports. "What's wrong with the codebase?" "Where are the security gaps?" Output is a document, not code changes. Discovery prompts live in `reusable/` (tech-debt-audit.md, security-audit.md).

**Execution** reads narrowly and produces code changes. "Change this specific thing in this specific file." Output is modified source files. Execution prompts live in task folders (`agents/<task-name>/`).

These must never be mixed. If an execution prompt contains the word "find" or "search for," it's doing discovery work. Discovery already found the problem — the execution prompt should say exactly what to change and where. This matters because discovery needs broad context (many files, cross-referencing) while execution needs narrow context (one task, few files). Mixing them wastes the context window on work that's already been done.

---

## How to Make a Code Change

### 1. Scaffold

```bash
bash agents/reusable/new-task.sh feature-name 3
```

Creates `agents/feature-name/` with PLAN.md, prompt.md, and 3 template files (01-TODO.md, 02-TODO.md, 03-TODO.md).

### 2. Write the prompts

Rename each template and fill it in. Every prompt has the same structure:

- **Context** — what exists, what this builds on, relevant file paths
- **Goal** — one sentence, what "done" looks like
- **Files to Modify** — explicit list, the agent should not touch anything else
- **Specific Changes** — precise instructions, code snippets if needed
- **DO NOT** — constraints and guardrails
- **Verify** — build commands and manual checks

Each prompt should produce 8–80 lines of meaningful changes across 1–3 files. If it's bigger than that, split it. If it's smaller, merge it with the next one.

Update PLAN.md to match the new filenames.

### 3. Run

```bash
bash agents/reusable/run-loop.sh agents/feature-name
```

Walk away. The script:
- Spawns a fresh Claude for each unchecked task in PLAN.md
- Kills the process when the task checkbox flips
- Moves to the next task with a clean context
- After all tasks complete, automatically runs the diff audit agent
- Writes the audit report to `agents/audits/YYYY-MM-DD-feature-name.md`

### 4. Review

Read the audit report. Three possible outcomes:

- **PASS** — review diffs, commit, push
- **PASS WITH WARNINGS** — review diffs, decide if warnings matter, commit
- **NEEDS FIXES** — scaffold a fix batch (`new-task.sh feature-name-fixes N`), write prompts addressing the specific violations, run the loop again

### 5. Archive

```bash
mv agents/feature-name agents/archive/feature-name
```

Add a row to `agents/archive/INDEX.md`.

---

## Standalone Audit

To audit changes outside the loop (manual edits, re-audit after fixes):

```bash
bash agents/reusable/run-audit.sh agents/feature-name
```

Same audit agent, same checks, writes to `agents/audits/`. Auto-increments the filename if you run it twice in one day.

---

## How the Loop Actually Works

The loop (`run-loop.sh`) is deliberately simple. It's a bash while loop that:

1. Checks PLAN.md for unchecked tasks — if none, exits
2. Pipes prompt.md to a fresh `claude --dangerously-skip-permissions` process
3. Polls PLAN.md every 5 seconds — when a new checkbox appears, kills the process
4. Sleeps 3 seconds, loops back to step 1

The kill-on-completion is the key mechanism. The execution agent checks off its task in PLAN.md when it thinks it's done. The harness sees the checkbox change and kills the process immediately. This prevents the agent from wandering — doing extra "cleanup," starting the next task uninvited, or running git commands.

### The Fake Git Wrapper

The script puts a fake `git` on the PATH that blocks all write operations (commit, push, add, checkout, etc.) but allows reads (status, diff, log, show, branch). This is harness-level enforcement — it doesn't matter if the prompt says "don't use git" because the prompt can be ignored. The fake wrapper can't be ignored.

### context.md

The driver prompt tells each iteration to read `context.md` at startup and append notes before finishing. This is how knowledge survives process kills. If task 02 discovers that a component was structured differently than expected, it writes that to context.md so task 03 doesn't repeat the discovery.

### caffeinate

Built into the script. Prevents macOS from sleeping during long runs.

---

## Key Principles

1. **The context window is an array.** Minimize what goes in it. One goal per window.
2. **Fresh context every iteration.** Kill the process. Never compact.
3. **The filesystem is shared memory.** PLAN.md is the state machine. context.md is the message bus.
4. **Back pressure belongs in the harness.** System-level enforcement (fake git, process kills) over prompt-level instructions ("please don't push").
5. **Separate discovery from execution.** Audit first, then fix. Never ask an execution agent to find things.
6. **Prompts are precise.** File paths, line numbers, specific changes. If the agent has to make judgment calls, the prompt is too vague.

---

## How Everything Connects

The system has three layers: knowledge, tooling, and output.

**Knowledge** (`initialization/`) feeds everything. CLAUDE.md tells the execution agent how to write code. The same CLAUDE.md tells the audit agent what to check against. README.md gives both agents a map of the codebase so they can find files without blind searching. workflow-architecture.md explains the system itself so a new session can adapt when something doesn't fit the template.

**Tooling** (`reusable/`) orchestrates execution. `new-task.sh` creates the task folder. `run-loop.sh` runs the prompts and triggers the audit. `run-audit.sh` runs the audit independently. The audit prompts (`diff-audit.md`, `tech-debt-audit.md`, `security-audit.md`) define what gets checked. None of these know anything about a specific feature — they're generic infrastructure.

**Output** goes to two places. `audits/` collects timestamped reports from every run — a persistent quality history. `archive/` collects completed task folders with an INDEX.md so you can search what was done.

The flow: knowledge feeds execution, execution produces output, output feeds review, review decides if you archive or fix. Nothing is circular. Each layer only reads from the layers above it.

---

## When NOT to Use the Loop

Not everything needs a 4-prompt batch. For quick one-off changes (fix a typo, add a CSS rule, rename a variable), just do it in a normal Claude session. The loop exists for multi-step work where context management matters — features that touch multiple files, refactors that need to happen in order, anything where you'd normally lose track of what's been done.

The rule of thumb: if you can describe the change in one prompt and verify it in under a minute, skip the loop. If it's 3+ prompts or you'd need to remember state between steps, use the loop.

---

## Further Reading

- `agents/initialization/workflow-architecture.md` — the deep dive: why compaction fails, how the pin works, the economics, hard-won lessons from real runs
- `agents/initialization/CLAUDE.md` — project conventions the audit agent checks against
- `agents/initialization/README.md` — the keyword-rich lookup table for finding files and routes
- `agents/archive/INDEX.md` — history of all completed batches

---

## Project Map

Quick reference so you don't have to grep. For the full version with every route, handler, schema, and keyword, see `agents/initialization/README.md`.

### Backend (Python/FastAPI, hexagonal architecture)

| Domain | Route Prefix | Service | Route File |
|--------|-------------|---------|------------|
| Contacts | `/today`, `/customer`, `/action`, `/import` | `services/contacts.py` | `adapters/driving/api/contacts.py` |
| Billing | `/checkout`, `/portal`, `/subscription`, `/webhook` | `services/billing.py` | `adapters/driving/api/billing.py` |
| Calendar | `/auth/google/*`, `/calendar/events`, `/sync/*` | `services/calendar.py` | `adapters/driving/api/calendar.py` |
| Estimates | `/estimates`, `/estimate/{token}` | `services/estimates.py` | `adapters/driving/api/estimates.py` |
| Team | `/invite`, `/{user_id}/role` | `services/team.py` | `adapters/driving/api/team.py` |
| Admin | `/health`, `/audit/logs`, `/metrics/kpi` | `services/admin.py` | `adapters/driving/api/admin.py` |
| Auth | `/me` | — | `adapters/driving/api/auth.py` |

All paths relative to `backend/app/`. Routes are thin — parse input, call service, return response. Business logic lives in services. DB access goes through `adapters/driven/database/repositories/`. External integrations (Stripe, Google, Twilio, Clerk, PostHog) through `adapters/driven/`.

### Frontend (React, Vite)

| Area | Key Components | Files |
|------|---------------|-------|
| Dashboard | TodayView, CustomerCard, ExpandedCard | `components/TodayView.jsx`, `CustomerCard.jsx`, `ExpandedCard.jsx` |
| Call workflow | ActionModal → CallInitiateStep → CallReachStep → OutcomeStep → ScheduleStep | `components/ActionModal.jsx`, `Call*.jsx`, `OutcomeStep.jsx`, `ScheduleStep.jsx` |
| Estimates | EstimatesView, SendEstimateModal, EstimateView (public) | `components/Estimates*.jsx`, `EstimateView.jsx` |
| Import | ImportView, InlineSpreadsheet, FileDropZone | `components/ImportView.jsx`, `InlineSpreadsheet.jsx` |
| Settings | Settings, CalendarSettings, TeamSettings, VisibilitySettings | `components/Settings.jsx`, `*Settings.jsx` |
| Landing | LandingPage, AboutPage, SignUpPage | `components/LandingPage.jsx`, `AboutPage.jsx` |
| Shared | SkeletonCard, ErrorBoundary, DarkModeToggle, LanguageToggle, GuidedTour | various |

All in `frontend/src/`. State management: `stores/customerStore.jsx`, `stores/uiStore.jsx`. Auth bridge: `contexts/ClerkAuthBridge.jsx`. CSS variables in `index.css`. Translations in `locales/{en,es}/translation.json`.

### Key conventions (abbreviated)

- **Hex architecture**: routes thin, logic in services, DB through repositories, external through adapters
- **Multi-tenancy**: every DB query filters by `account_id`
- **Auth**: `get_current_user` on all endpoints, `require_active_subscription` on write endpoints
- **Frontend styles**: use CSS variables from `index.css`, no inline styles except dynamic values
- **Translations**: all user-facing strings in both `en/` and `es/` locale files via `useTranslation()`
- **Naming**: `handle*` for event handlers, `on*` for callback props

For the complete conventions list, read `agents/initialization/CLAUDE.md`.
