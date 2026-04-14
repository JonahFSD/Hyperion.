# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development
```bash
make dev              # Start full stack (docker-compose: postgres, redis, backend, frontend)
make dev-down         # Stop development environment
make dev-logs         # Tail all service logs
```

Frontend: http://localhost:5173 | Backend: http://localhost:8000 | API Docs: http://localhost:8000/docs

### Running without Docker
```bash
# Backend (from backend/)
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (from frontend/)
npm install && npm run dev
```

### Testing (backend only — no frontend tests)
```bash
make test                 # All backend tests
make test-fast            # Stop on first failure (-x --tb=short)
make test-coverage        # With HTML coverage report

# Single test file
cd backend && python -m pytest tests/test_auth.py -v

# Single test function
cd backend && python -m pytest tests/test_auth.py::test_function_name -v

# By marker
cd backend && python -m pytest tests/ -m security -v
```

Markers: `slow`, `integration`, `security`

**Test setup**: Tests use SQLite in-memory with FastAPI dependency overrides. Auth is shimmed with HS256 JWTs (instead of Clerk JWKS). Fixtures in `backend/tests/conftest.py` provide `client`, `test_db`, `auth_headers`, `admin_auth_headers`, `member_auth_headers`, `other_auth_headers` (second tenant), and a `TestDataFactory`.

### Linting
```bash
ruff check backend/       # Python linter (CI uses this)
ruff check backend/ --fix # Auto-fix
bandit -r backend/ -ll -ii -x "backend/tests/*,backend/venv/*"  # Security linter
```

### Database Migrations (via Docker)
```bash
make migrate-dev                          # Apply all migrations
make migrate-new NAME='description'       # Generate new migration
make migrate-history                      # View migration history
make migrate-downgrade                    # Rollback one step
```

### CI Pipeline
GitHub Actions runs on push to `main` and all PRs:
- **Backend job**: `ruff check backend/` then `pytest backend/tests/`
- **Frontend job**: `npm ci` then `npm run build`

Deployment is separate — Vercel (frontend) and Railway (backend) auto-deploy from GitHub.

---

## Architecture

### Backend: Hexagonal Architecture (FastAPI + Python 3.11)

Entry point: `backend/main.py`

```
backend/app/
├── adapters/
│   ├── driving/api/       # Route handlers (thin: parse input → call service → map to HTTP)
│   │   ├── contacts.py, notes.py, calendar.py, billing.py
│   │   ├── team.py, admin.py, auth.py, demo.py, estimates.py
│   └── driven/            # Concrete implementations
│       ├── database/
│       │   ├── models.py      # SQLAlchemy ORM models
│       │   ├── session.py     # Engine, SessionLocal, get_db()
│       │   └── repositories/  # Repository implementations
│       ├── stripe/, clerk/, google/, email/, twilio/
├── domain/                # Business entities & exceptions
├── ports/                 # Abstract interfaces (repository contracts)
└── services/              # All business logic (constructor-injected dependencies)
```

**Key patterns:**
- Multi-tenancy: all queries filtered by `account_id`
- Auth: Clerk JWT validation via JWKS
- DB: PostgreSQL (prod) / SQLite (dev, file: `crm.db`)
- Rate limiting: slowapi with per-endpoint limits
- Config: Pydantic Settings v2 in `backend/config.py`, loaded from `.env`

### Frontend: React 18 + Vite

Entry point: `frontend/src/main.jsx` → `App.jsx`

- **Tab navigation** within app via RouterContext (today, calendar, estimates, settings)
- **Page routing** between public pages via `window.location.href` (not React Router)
- **Auth**: ClerkProvider wraps app; ClerkAuthBridge context for auth state
- **i18n**: i18next with EN/ES locales in `frontend/src/locales/{en,es}/translation.json`
- **State**: Component state + stores in `frontend/src/stores/` (customerStore, uiStore)
- **Analytics**: PostHog

### External Services
Clerk (auth), Stripe (billing), Google (calendar/contacts), Twilio (SMS estimates), Resend (email), Sentry (errors)

### Dependency Injection
`backend/app/adapters/driving/api/dependencies.py` builds the full dependency graph per-request via FastAPI's `Depends()`. Service factories wire ports to driven adapters so route handlers never instantiate dependencies directly.

### Feature → Implementation Map

| Feature | Backend | Frontend |
|---------|---------|----------|
| Priority scoring | `services/scoring.py` + `services/recommendations.py` | `TodayView.jsx` (ranked list, red/amber/green) |
| Call workflow | — | `ActionModal.jsx` → 4-step wizard (initiate → reach → outcome → schedule) |
| SMS estimates | `services/estimates.py` + `TwilioAdapter` | `SendEstimateModal.jsx`, public view at `/estimate/:token` |
| Google Calendar sync | `services/calendar.py` + `GoogleCalendarProvider` | `CalendarView.jsx` |
| CSV import/export | contacts service (validate, dedupe, enforce limits) | `ImportView.jsx` + `InlineSpreadsheet.jsx` + `csvParser.js` |
| Billing / paywall | `services/billing.py` + `StripePaymentProvider` | `SubscriptionGate.jsx` (past-due = read-only) |
| Team management | `services/team.py` (invite via Resend, RBAC, audit logs) | Settings tab |
| Onboarding | — | `GuidedTour.jsx` + `ProgressivePrompts.jsx` + `OnboardingScreen.jsx` |
| Demo mode | `services/demo_data.py` | `DemoView.jsx` + `DemoBanner.jsx` |
| KPI dashboard | — | `KpiDashboard.jsx` |

---

## Workflow Principles

- **Specs-first**: Check `agents/initialization/README.md` lookup table before grepping.
- **One goal per session**, small changes (1-3 files, 8-80 lines)
- `agents/archive/` — DO NOT read during implementation

---

## Code Style

### Backend
- Follow hexagonal layering: routes are thin, logic in services, DB access through ports
- All new queries must filter by `account_id`

### Frontend
- Use existing CSS variables from `frontend/src/index.css` (`--bg-surface`, `--accent-blue`, `--space-*`, `--color-red/amber/green`)
- Add styles to component-specific CSS files, NOT to index.css
- Add translations to both `en/` and `es/` locale files
- Naming: `handle*` for event handlers, `on*` for callback props
- No new dependencies without explicit approval
- No inline styles except for dynamic/computed values

### Styling Quick Reference
- Buttons: `.btn` base + `.btn-call` (blue) / `.btn-schedule` (green) / `.btn-secondary` (muted)
- Corners: `var(--radius-sm)` — never hardcode hex colors, always use CSS variables
- Typography: titles 28px bold, section headers 11px uppercase, body 15px

### Navigation
```jsx
// Page transitions (landing/demo/about)
window.location.href = '/target-path'
// In-app tab switching uses RouterContext — don't mix the two
```

---

## Adding Features Checklist

1. Check if similar pattern exists in codebase
2. Use existing components/utilities before creating new ones
3. Add translations (both EN and ES)
4. Scope CSS to component file, not index.css
5. No new dependencies without approval
6. Test dark mode appearance
7. Consider mobile layout (hamburger menu context)

---

## Key File Locations

| Purpose | Location |
|---------|----------|
| Backend entry point | `backend/main.py` |
| Backend config | `backend/config.py` |
| DB models | `backend/app/adapters/driven/database/models.py` |
| DB session | `backend/app/adapters/driven/database/session.py` |
| Migrations | `backend/migrations/versions/` |
| Frontend entry | `frontend/src/main.jsx` → `frontend/src/App.jsx` |
| Landing page | `frontend/src/components/LandingPage.jsx` |
| Demo page | `frontend/src/components/DemoView.jsx` |
| CSS variables | `frontend/src/index.css` (lines 1-140) |
| Translations | `frontend/src/locales/{en,es}/translation.json` |
| API module | `frontend/src/api.js` |
| Dependency injection | `backend/app/adapters/driving/api/dependencies.py` |
| Test fixtures | `backend/tests/conftest.py` |
| Specs lookup | `agents/initialization/README.md` |
