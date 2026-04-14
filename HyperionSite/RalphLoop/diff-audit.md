# Reusable Diff Audit Prompt

> Runs automatically after the Ralph loop completes all tasks.
> Audits the uncommitted changes against project conventions and task specs.
> Produces a structured report at `agents/audits/YYYY-MM-DD-<task-label>.md`.

---

## Instructions

You are auditing uncommitted code changes produced by an autonomous AI coding loop. Your job is to review the diff, check it against project conventions and the task specifications, and produce a structured report.

**Do NOT modify any source files. This is a read-only audit. Your only output is the report.**

### Step 1: Gather context

1. Run `git diff --stat` to see which files changed and the scope of changes
2. Run `git diff` to read the full diff
3. Read `agents/initialization/CLAUDE.md` for project conventions
4. Read all task prompt files in the task folder (find them via PLAN.md)
5. If backend files changed, read `agents/initialization/README.md` for architecture reference

### Step 2: Check conventions

Evaluate the diff against every applicable rule below. Flag each violation with a severity.

#### Backend conventions (if backend files changed)
- **Hexagonal layering**: Routes must be thin (parse input → call service → return response). Business logic must not live in route handlers. DB access must go through ports/repositories, never direct queries in services or routes.
- **Multi-tenancy**: Every new database query must filter by `account_id`. Check for queries that could leak data across tenants.
- **Auth gates**: New endpoints must have appropriate authentication (`get_current_user`) and authorization (role checks where needed). New write endpoints must have `require_active_subscription`.
- **Input validation**: New request inputs must have Pydantic schema validation with field constraints (max lengths, enums, regex patterns where appropriate).
- **Error handling**: Exceptions must not leak internal details (file paths, stack traces, database IDs) to the client.

#### Frontend conventions (if frontend files changed)
- **CSS variables**: No hardcoded hex colors. Must use CSS variables from `frontend/src/index.css` (`--bg-surface`, `--accent-blue`, `--color-red`, `--color-amber`, `--color-green`, `--radius-sm`, etc.). Exception: landing page may use its own scoped color values for dark-mode-only styles.
- **Styles location**: New styles must be in component-specific CSS files, NOT in `index.css`.
- **Translations**: Any new user-facing strings must be added to BOTH `frontend/src/locales/en/translation.json` AND `frontend/src/locales/es/translation.json`. Use `useTranslation()` hook, not hardcoded strings. Exception: landing page marketing copy may be hardcoded English.
- **Naming conventions**: Event handlers must use `handle*` prefix. Callback props must use `on*` prefix.
- **No inline styles**: Except for dynamic/computed values (e.g., `style={{ transitionDelay }}` is OK, `style={{ color: 'red' }}` is not).
- **No new dependencies**: Check `package.json` for additions. Flag any new dependency.
- **Dark mode**: If the component renders in dark mode context, verify text readability and contrast.

#### General conventions (always check)
- **Scope creep**: Compare files actually modified against the "Files to Modify" list in each task prompt. Flag any files touched that weren't specified.
- **Line count**: Each task prompt should produce roughly 8-80 lines of meaningful changes. Flag tasks that produced significantly more (may indicate the prompt was too broad).
- **Unused code**: Check for unused imports, unreachable code, console.log statements left in, commented-out code blocks.
- **No git operations**: Verify no `.git` changes, no new `.gitignore` entries unless specified.

#### Security (always check)
- **New endpoints**: Must have auth middleware. Must not expose internal IDs unnecessarily.
- **User input**: Must be validated/sanitized before use. Check for XSS vectors in rendered content.
- **Secrets**: No API keys, tokens, passwords, or connection strings in source files.
- **account_id filtering**: Any new database access must enforce tenant isolation.

### Step 3: Check spec compliance

For each task prompt in PLAN.md:

1. Read the prompt's **Goal** section. Did the changes achieve it?
2. Read the prompt's **Specific Changes** section. Were all steps implemented?
3. Read the prompt's **DO NOT** section. Were any constraints violated?
4. Run the prompt's **Verify** commands. Do they pass?
5. Check if the prompt's stated files match the actually modified files.

### Step 4: Write the report

Write the report to the path provided in your instructions (typically `agents/audits/YYYY-MM-DD-<task-label>.md`). The run-loop.sh script passes this path explicitly.

Use this format:

```markdown
# Diff Audit Report — [task folder name]
**Date:** YYYY-MM-DD
**Files changed:** N files, +X -Y lines

## Summary
[1-2 sentence overall assessment: PASS / PASS WITH WARNINGS / NEEDS FIXES]

## Convention Violations

### [SEVERITY] — [short description]
- **File:** [path:line]
- **Rule:** [which convention was violated]
- **Details:** [what's wrong and what it should be]

(repeat for each violation, or "None found." if clean)

## Spec Compliance

### Prompt: [filename]
- **Goal met:** Yes/No
- **All steps implemented:** Yes/No
- **DO NOT constraints respected:** Yes/No
- **Verify commands:** Pass/Fail
- **Notes:** [anything notable]

(repeat for each prompt)

## Security Review
[Any security concerns found in the diff, or "No security issues found."]

## Recommendations
[Ordered list of things to fix before committing, if any. Or "Ready to commit." if clean.]
```

## DO

- Be thorough — check every changed line against conventions
- Run the verify commands from each prompt
- Note positive things too (good patterns followed, clean implementation)
- Be specific with line numbers when flagging issues

## DO NOT

- Do not modify any source files
- Do not run git add, git commit, or git push
- Do not read files in `agents/archive/`
- Do not attempt to fix issues — only report them
