# Tech Debt Audit Prompt

> Run this in Claude Code CLI from the CallSheet3 project root.
> This is a multi-step audit — run each section as a separate prompt, or run the full thing if your context window is large enough.

---

## Full Audit Prompt

```
## Context

I need a comprehensive tech debt audit of the CallSheet CRM codebase. This is a React frontend + Python/FastAPI backend using hexagonal architecture. The project conventions are documented in CLAUDE.md at the project root.

Evaluate every category below. For each one, provide:
- A severity rating: 🔴 CRITICAL, 🟠 MAJOR, 🟡 MINOR, ✅ CLEAN
- Exact counts and file paths
- The top 3 worst offenders with line numbers
- A concrete fix recommendation (1-2 sentences)

Output the results as a markdown file saved to `specs/tech-debt-audit-YYYY-MM-DD.md`.

---

### CATEGORY 1: Component Complexity

Audit criteria:
- [ ] Flag any .jsx file over 300 lines (threshold from CLAUDE.md: 8-80 lines per task)
- [ ] Flag any component with more than 8 useState hooks
- [ ] Flag any component with more than 3 useEffect hooks
- [ ] Flag any component that manages 2+ unrelated concerns (e.g. pagination AND modals AND localStorage)
- [ ] Count useRef instances used as mutable flags (anti-pattern vs. proper state)

Known issue: `TodayView.jsx` has 18 useState hooks and manages priority pagination, all-customers pagination, search, modals, and localStorage tracking all in one component.

### CATEGORY 2: Duplicate Systems

Audit criteria:
- [ ] Identify features implemented more than once with different approaches
- [ ] Count total lines across all implementations of duplicated features
- [ ] Assess which implementation is most complete/maintainable

Known issue: There are 4 separate tour/onboarding systems:
  - `GuidedTour.jsx` (~379 lines) — custom DOM-based
  - `TutorialWizard.jsx` (~372 lines) — step wizard
  - `TutorialTour.jsx` (~195 lines) — react-joyride based
  - `OnboardingTour.jsx` — first-time flow
Plus `react-joyride` (45KB) as a dependency alongside 3 custom implementations.

### CATEGORY 3: CSS Architecture

Audit criteria:
- [ ] Count total lines in `frontend/src/index.css` (known: ~3000 lines)
- [ ] Identify CSS classes defined in index.css that should be in component-specific .css files (per CLAUDE.md: "Don't add styles to index.css")
- [ ] Grep each class in index.css for usage — estimate % unused
- [ ] Identify sections that could be split into separate files (landing page styles, auth page styles, calendar overrides)
- [ ] Find dark mode overrides (`[data-theme="dark"]`) that duplicate light mode values without changes
- [ ] Count redundant CSS variable aliases (multiple variables mapping to same value)

### CATEGORY 4: Inline Styles

Audit criteria:
- [ ] Count all `style={{` occurrences across .jsx files
- [ ] Group by file, sorted by count (known: ~64 total, ExpandedCard.jsx worst at 12)
- [ ] Identify repeated inline style patterns (same style object 3+ times = should be a class)
- [ ] Separate legitimate dynamic styles (computed values) from lazy inline styles (static values)
- [ ] Per CLAUDE.md: "Don't use inline styles except for computed/dynamic values"

### CATEGORY 5: i18n Coverage

Audit criteria:
- [ ] Find all raw English strings in JSX that aren't wrapped in `t()` calls
- [ ] Check for hardcoded strings in: button labels, headings, placeholder text, error messages, aria-labels
- [ ] Known issues: "ESTIMATES" hardcoded in App.jsx lines 285 and 335
- [ ] Check SendEstimateModal.jsx, SignUpPage.jsx, DemoBanner.jsx, SyncHistory.jsx, InlineSpreadsheet.jsx for untranslated strings
- [ ] Verify all translation keys in en.json have corresponding es.json entries
- [ ] Find orphaned translation keys (defined but never referenced in code)

### CATEGORY 6: localStorage Misuse

Audit criteria:
- [ ] List every localStorage.getItem and localStorage.setItem call with file, line, and purpose
- [ ] Flag cases where localStorage is used to pass data between components (should be props/context)
- [ ] Flag user-specific data stored only in localStorage that should be server-side (onboarding state, A/B test assignment, tour progress)
- [ ] Known issue: `crm_spreadsheet_data` in localStorage used as inter-component communication (App.jsx → InlineSpreadsheet.jsx)
- [ ] Known issue: A/B test group assignment (`tour_ab_group`) resets on new device/cache clear
- [ ] Assess: which localStorage keys are appropriate (theme, language) vs. inappropriate (user progress, data)

### CATEGORY 7: Error Handling

Audit criteria:
- [ ] Find all `.catch` handlers in frontend code — categorize as "shows error to user" vs. "silently swallows"
- [ ] Find promise chains with NO .catch at all
- [ ] Check if ErrorBoundary component exists and is actually wired up in the component tree
- [ ] Assess api.js error handling: are custom error types (ApiError, AuthenticationError, etc.) caught and handled distinctly in components, or are they all treated the same?
- [ ] Known issue: TodayView.jsx `.catch(() => setPriorityHasMore(false))` silently hides API failures
- [ ] Check for any global error notification system (toast/snackbar) — does one exist?

### CATEGORY 8: Prop Drilling & State Architecture

Audit criteria:
- [ ] Map the deepest prop-passing chains (per CLAUDE.md: "No prop drilling beyond 2 levels")
- [ ] Count props passed to each top-level component from App.jsx
- [ ] Identify callback props threaded through intermediary components that don't use them
- [ ] Assess whether React Context is underutilized (currently only ClerkAuthBridge)
- [ ] Check if any state management library (Zustand, Redux) exists but is barely used

### CATEGORY 9: API Layer Consistency

Audit criteria:
- [ ] Verify ALL API calls go through `frontend/src/api.js` (no direct fetch/axios in components)
- [ ] Check for hardcoded API URLs anywhere
- [ ] Assess error handling consistency in api.js — are all endpoints handled the same way?
- [ ] Check for missing request/response typing (no TypeScript)
- [ ] Look for API calls that should have loading/error states in UI but don't

### CATEGORY 10: Backend Architecture Compliance

Audit criteria:
- [ ] Verify hexagonal architecture per CLAUDE.md: routes (driving adapters) should be thin, services hold business logic, repositories are ports
- [ ] Flag any route handler over 30 lines (may contain business logic)
- [ ] Check if service layer exists and is properly separated from routes
- [ ] Known concern: `backend/app/adapters/driving/api/contacts.py` is 325 lines — likely has business logic in route handlers
- [ ] Check for direct database access in route handlers (should go through repositories)
- [ ] Verify dependency injection pattern is consistent

### CATEGORY 11: Bundle & Dependencies

Audit criteria:
- [ ] Read package.json — flag packages over 100KB
- [ ] Identify multiple packages serving the same purpose
- [ ] Known: xlsx (700KB) is the heaviest dependency
- [ ] Known: react-joyride (45KB) coexists with 3 custom tour implementations
- [ ] Check for packages in dependencies that should be in devDependencies
- [ ] Assess if any heavy packages could be replaced with lighter alternatives or lazy-loaded

### CATEGORY 12: Dead Code

Audit criteria:
- [ ] Find .jsx components that are defined but never imported anywhere
- [ ] Find exported functions in api.js that no component calls
- [ ] Find CSS classes defined but never referenced in any .jsx file
- [ ] Check for commented-out code blocks (>5 lines)
- [ ] Find translation keys with no corresponding `t('key')` usage in code

---

## Output Format

Save the audit to `specs/tech-debt-audit-YYYY-MM-DD.md` with:

1. **Executive Summary** — 3-sentence overall health assessment
2. **Scorecard Table** — all 12 categories with severity rating and 1-line finding
3. **Detailed Findings** — each category with full evidence
4. **Priority Action Items** — ranked list of fixes, grouped by:
   - 🔴 Fix Now (blocks maintainability or causes bugs)
   - 🟠 Fix Soon (accumulating cost)
   - 🟡 Fix When Convenient (nice-to-have cleanup)
5. **Estimated Effort** — rough t-shirt sizes (S/M/L) for each action item

## DO

- Use exact file paths and line numbers
- Show code snippets for the worst offenders
- Be specific about what "fix" means for each item
- Count things — don't say "many" when you can say "64"

## DO NOT

- Modify any files — this is read-only audit
- Suggest adding new dependencies without noting the tradeoff
- Suggest TypeScript migration as a standalone task (it's too large)
- Include items that are already following best practices — only report problems
```

---

## Breaking It Up (Optional)

If the full audit is too large for one session, run these as individual prompts in order:

1. **Frontend complexity** — Categories 1, 2, 4, 8
2. **Styling & i18n** — Categories 3, 5
3. **Data & error handling** — Categories 6, 7, 9
4. **Backend & bundle** — Categories 10, 11, 12

Each sub-prompt should still output to the same `specs/tech-debt-audit-YYYY-MM-DD.md` file (append mode).
