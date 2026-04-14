# CallSheet Specs — Consolidated Reference

Quick reference for Claude Code to find relevant specs and source files.
Search by feature, keyword, or file type.

---

## Table of Contents

1. [Backend Routes](#backend-routes)
2. [Backend Services](#backend-services)
3. [Backend Utilities](#backend-utilities)
4. [Backend Schemas](#backend-schemas)
5. [Backend Dependencies (DI)](#backend-dependencies-di)
6. [Frontend Components](#frontend-components)
7. [Frontend State](#frontend-state)
8. [Frontend Resources](#frontend-resources)
9. [Architecture Layers](#architecture-layers)
10. [SQLAlchemy Models](#sqlalchemy-models)
11. [Dependency Graph](#dependency-graph)
12. [Workflow & Automation](#workflow--automation)
13. [Security Findings (02/14/26)](#security-findings-021426)

---

## Backend Routes

### Contacts Domain
| Route | Method | Handler | File |
|-------|--------|---------|------|
| `/today` | GET | `get_today_dashboard` | `backend/app/adapters/driving/api/contacts.py` |
| `/customer/{id}` | GET | `get_customer_detail` | `backend/app/adapters/driving/api/contacts.py` |
| `/action` | POST | `record_action` | `backend/app/adapters/driving/api/contacts.py` |
| `/import` | POST | `import_customers` | `backend/app/adapters/driving/api/contacts.py` |
| `/import/template` | GET | `get_import_template` | `backend/app/adapters/driving/api/contacts.py` |
| `/demo-customers` | POST | `load_demo_customers` | `backend/app/adapters/driving/api/contacts.py` |
| `/customers/clear-all` | DELETE | `clear_all_customers` | `backend/app/adapters/driving/api/contacts.py` |
| `/customer/{id}/notes` | GET | `get_notes` | `backend/app/adapters/driving/api/contacts.py` |
| `/customer/{id}/notes` | POST | `create_note` | `backend/app/adapters/driving/api/contacts.py` |
| `/notes/{id}` | PATCH | `update_note` | `backend/app/adapters/driving/api/contacts.py` |
| `/notes/{id}` | DELETE | `delete_note` | `backend/app/adapters/driving/api/contacts.py` |

**Keywords:** customer, contact, today, priority, score, action, call, schedule, dismiss, import, csv, notes, pin

### Billing Domain
| Route | Method | Handler | File |
|-------|--------|---------|------|
| `/checkout` | POST | `create_checkout` | `backend/app/adapters/driving/api/billing.py` |
| `/portal` | POST | `create_portal` | `backend/app/adapters/driving/api/billing.py` |
| `/subscription` | GET | `get_subscription` | `backend/app/adapters/driving/api/billing.py` |
| `/cancel` | POST | `cancel_subscription` | `backend/app/adapters/driving/api/billing.py` |
| `/reactivate` | POST | `reactivate_subscription` | `backend/app/adapters/driving/api/billing.py` |
| `/upgrade` | POST | `upgrade_subscription` | `backend/app/adapters/driving/api/billing.py` |
| `/webhook` | POST | `handle_webhook` | `backend/app/adapters/driving/api/billing.py` |

**Keywords:** billing, stripe, checkout, payment, subscribe, subscription, portal, cancel, webhook, tier

### Calendar Domain (Google)
| Route | Method | Handler | File |
|-------|--------|---------|------|
| `/auth/google/connect` | GET | `connect_google` | `backend/app/adapters/driving/api/calendar.py` |
| `/auth/google/callback` | GET | `google_oauth_callback` | `backend/app/adapters/driving/api/calendar.py` |
| `/auth/google/status` | GET | `get_google_status` | `backend/app/adapters/driving/api/calendar.py` |
| `/auth/google/disconnect` | POST | `disconnect_google` | `backend/app/adapters/driving/api/calendar.py` |
| `/calendar/events` | GET | `get_calendar_events` | `backend/app/adapters/driving/api/calendar.py` |
| `/google/contacts-preview` | GET | `preview_google_contacts` | `backend/app/adapters/driving/api/calendar.py` |
| `/google/import-contacts` | POST | `import_google_contacts` | `backend/app/adapters/driving/api/calendar.py` |
| `/sync/history` | GET | `get_sync_history` | `backend/app/adapters/driving/api/calendar.py` |
| `/sync/{sync_id}` | GET | `get_sync_detail` | `backend/app/adapters/driving/api/calendar.py` |

**Keywords:** google, calendar, oauth, connect, sync, gcal, contacts, import, events

### Estimates Domain
| Route | Method | Handler | File |
|-------|--------|---------|------|
| `/estimates` | POST | `create_estimate` | `backend/app/adapters/driving/api/estimates.py` |
| `/estimates/{id}/send` | POST | `send_estimate` | `backend/app/adapters/driving/api/estimates.py` |
| `/estimate/{token}` | GET | `get_public_estimate` | `backend/app/adapters/driving/api/estimates.py` |

**Keywords:** estimate, sms, twilio, quote, send, public, token

### Team Domain
| Route | Method | Handler | File |
|-------|--------|---------|------|
| `/` | GET | `list_members` | `backend/app/adapters/driving/api/team.py` |
| `/invite` | POST | `invite_member` | `backend/app/adapters/driving/api/team.py` |
| `/{user_id}/role` | PUT | `update_role` | `backend/app/adapters/driving/api/team.py` |
| `/{user_id}` | DELETE | `remove_member` | `backend/app/adapters/driving/api/team.py` |

**Keywords:** team, member, invite, role, owner, admin, permission, rbac

### Admin Domain
| Route | Method | Handler | File |
|-------|--------|---------|------|
| `/health` | GET | `health_check` | `backend/app/adapters/driving/api/admin.py` |
| `/health/full` | GET | `full_health_check` | `backend/app/adapters/driving/api/admin.py` |
| `/audit/logs` | GET | `get_audit_logs` | `backend/app/adapters/driving/api/admin.py` |
| `/metrics/kpi` | GET | `get_kpi_metrics` | `backend/app/adapters/driving/api/admin.py` |

**Keywords:** admin, health, audit, log, kpi, metrics, dashboard

### Auth Domain
| Route | Method | Handler | File |
|-------|--------|---------|------|
| `/me` | GET | `get_me` | `backend/app/adapters/driving/api/auth.py` |

**Keywords:** auth, me, current_user, session, jwt, clerk

### Demo
| Route | Method | Handler | File |
|-------|--------|---------|------|
| `/customers` | GET | `get_demo_customers` | `backend/app/adapters/driving/api/demo.py` |

**Keywords:** demo, preview, sample

---

## Backend Services

| Service | File | Keywords |
|---------|------|----------|
| ContactsService | `backend/app/services/contacts.py` | customer, contact, crud |
| NotesService | `backend/app/services/notes.py` | note, pin, content |
| ScoringService | `backend/app/services/scoring.py` | score, priority, 0-100, red, yellow, green |
| RecommendationsService | `backend/app/services/recommendations.py` | recommend, suggest, next |
| BillingService | `backend/app/services/billing.py` | billing, stripe, subscription |
| CalendarService | `backend/app/services/calendar.py` | calendar, google, oauth, sync |
| TeamService | `backend/app/services/team.py` | team, invite, role |
| AdminService | `backend/app/services/admin.py` | admin, audit, health, kpi |
| EstimateService | `backend/app/services/estimates.py` | estimate, sms, twilio, quote |
| DemoDataService | `backend/app/services/demo_data.py` | demo, seed, sample |

---

## Backend Utilities

| File | Purpose | Keywords |
|------|---------|----------|
| `backend/clerk_auth.py` | JWT validation, user resolution | jwt, token, clerk, validate, auth |
| `backend/clerk_client.py` | Clerk API client | clerk, api, user, fetch |
| `backend/clerk_webhook.py` | Clerk user lifecycle webhooks | webhook, clerk, user.created, hmac |
| `backend/config.py` | Pydantic settings config | config, settings, env, environment |
| `backend/encryption.py` | Fernet encryption for OAuth tokens | encrypt, decrypt, fernet, token |
| `backend/logging_config.py` | Structured JSON logging | log, logging, json, request_id |
| `backend/main.py` | FastAPI entrypoint | app, fastapi, middleware, router |
| `backend/security.py` | OWASP security utilities | security, rate, limit, sanitize, csv, headers |

---

## Backend Schemas

All request/response models in `backend/app/adapters/driving/api/schemas.py`:

| Schema | Type |
|--------|------|
| `ActionRequest` | Request |
| `NoteCreateRequest` / `NoteUpdateRequest` | Request |
| `ImportContactsRequest` | Request |
| `CheckoutRequest` / `PortalRequest` | Request |
| `InviteMemberRequest` / `UpdateRoleRequest` | Request |
| `CustomerCardResponse` / `CustomerDetailResponse` | Response |
| `NoteResponse` / `ImportResponse` | Response |
| `CalendarEventResponse` / `ContactPreviewResponse` / `SyncRecordResponse` | Response |
| `CheckoutResponse` / `PortalResponse` | Response |
| `TeamMemberResponse` / `AuditLogResponse` | Response |
| `KPIResponse` / `HealthResponse` | Response |

**Keywords:** schema, pydantic, request, response, validation, model

---

## Backend Dependencies (DI)

All in `backend/app/adapters/driving/api/dependencies.py`:

| Dependency | Returns |
|------------|---------|
| `require_subscription` | Subscription check (allows read for past_due) |
| `require_active_subscription` | Active subscription check (write ops) |
| `get_contacts_service` | ContactsService |
| `get_notes_service` | NotesService |
| `get_calendar_service` | CalendarService |
| `get_billing_service` | BillingService |
| `get_team_service` | TeamService |
| `get_admin_service` | AdminService |
| `get_estimate_service` | EstimateService |
| `get_analytics` | PostHogAnalyticsProvider (singleton) |

**Keywords:** dependency, inject, service, require, get

---

## Frontend Components

| Component | File | Keywords |
|-----------|------|----------|
| TodayView | `frontend/src/components/TodayView.jsx` | today, dashboard, list, priority |
| CustomerCard | `frontend/src/components/CustomerCard.jsx` | card, customer, display |
| ExpandedCard | `frontend/src/components/ExpandedCard.jsx` | detail, expanded, timeline |
| ActionModal | `frontend/src/components/ActionModal.jsx` | action, modal, call, schedule |
| CallInitiateStep | `frontend/src/components/CallInitiateStep.jsx` | call, initiate, step 1 |
| CallReachStep | `frontend/src/components/CallReachStep.jsx` | call, reach, step 2 |
| OutcomeStep | `frontend/src/components/OutcomeStep.jsx` | outcome, step 3 |
| ScheduleStep | `frontend/src/components/ScheduleStep.jsx` | schedule, step 4 |
| DismissStep | `frontend/src/components/DismissStep.jsx` | dismiss, skip |
| ImportView | `frontend/src/components/ImportView.jsx` | import, csv, upload |
| InlineSpreadsheet | `frontend/src/components/InlineSpreadsheet.jsx` | spreadsheet, inline, data |
| ContactsReview | `frontend/src/components/ContactsReview.jsx` | review, contacts, google |
| CalendarView | `frontend/src/components/CalendarView.jsx` | calendar, events, view |
| CalendarSettings | `frontend/src/components/CalendarSettings.jsx` | calendar, google, settings |
| EstimatesView | `frontend/src/components/EstimatesView.jsx` | estimates, list |
| EstimateView | `frontend/src/components/EstimateView.jsx` | estimate, public, view |
| SendEstimateModal | `frontend/src/components/SendEstimateModal.jsx` | estimate, send, modal |
| Settings | `frontend/src/components/Settings.jsx` | settings, preferences |
| TeamSettings | `frontend/src/components/TeamSettings.jsx` | team, settings, invite |
| TeamMemberRow | `frontend/src/components/TeamMemberRow.jsx` | team, member, row |
| InviteForm | `frontend/src/components/InviteForm.jsx` | invite, form |
| SubscriptionGate | `frontend/src/components/SubscriptionGate.jsx` | gate, paywall, subscription |
| VisibilitySettings | `frontend/src/components/VisibilitySettings.jsx` | visibility, toggle |
| SyncHistory | `frontend/src/components/SyncHistory.jsx` | sync, history, google |
| KpiDashboard | `frontend/src/components/KpiDashboard.jsx` | kpi, metrics, analytics |
| LandingPage | `frontend/src/components/LandingPage.jsx` | landing, public, marketing |
| AboutPage | `frontend/src/components/AboutPage.jsx` | about, info |
| SignUpPage | `frontend/src/components/SignUpPage.jsx` | signup, register |
| SignupModal | `frontend/src/components/SignupModal.jsx` | signup, register, modal |
| DemoView | `frontend/src/components/DemoView.jsx` | demo, preview |
| DemoBanner | `frontend/src/components/DemoBanner.jsx` | demo, banner |
| OnboardingScreen | `frontend/src/components/OnboardingScreen.jsx` | onboard, welcome, new |
| GuidedTour | `frontend/src/components/GuidedTour.jsx` | tour, guide, help |
| ProgressivePrompts | `frontend/src/components/ProgressivePrompts.jsx` | progressive, prompts, onboard |
| EmptyStateWithCTA | `frontend/src/components/EmptyStateWithCTA.jsx` | empty, cta, placeholder |
| SkeletonCard | `frontend/src/components/SkeletonCard.jsx` | skeleton, loading |
| ErrorBoundary | `frontend/src/components/ErrorBoundary.jsx` | error, boundary, catch |
| ErrorFallback | `frontend/src/components/ErrorFallback.jsx` | error, fallback |
| DarkModeToggle | `frontend/src/components/DarkModeToggle.jsx` | dark, theme, toggle |
| LanguageToggle | `frontend/src/components/LanguageToggle.jsx` | language, i18n, locale |
| HelpButton | `frontend/src/components/HelpButton.jsx` | help, support |
| FeatureTooltip | `frontend/src/components/FeatureTooltip.jsx` | tooltip, help, hint |
| FileDropZone | `frontend/src/components/FileDropZone.jsx` | file, drop, upload |
| ImportGuidelines | `frontend/src/components/ImportGuidelines.jsx` | import, guidelines |

---

## Frontend State

| Store | File | Keywords |
|-------|------|----------|
| customerStore | `frontend/src/stores/customerStore.jsx` | customer, store, state |
| uiStore | `frontend/src/stores/uiStore.jsx` | ui, modal, view, store |
| ClerkAuthBridge | `frontend/src/contexts/ClerkAuthBridge.jsx` | auth, clerk, context, bridge |

---

## Frontend Resources

| Resource | File | Keywords |
|----------|------|----------|
| CSS Variables | `frontend/src/index.css` | css, variable, color, theme, style |
| English translations | `frontend/src/locales/en/translation.json` | i18n, english, translate |
| Spanish translations | `frontend/src/locales/es/translation.json` | i18n, spanish, translate |
| API client | `frontend/src/api.js` | api, fetch, request, http |

---

## Architecture Layers

| Layer | Path Pattern | Keywords |
|-------|--------------|----------|
| Driving adapters (API) | `backend/app/adapters/driving/api/*.py` | route, endpoint, handler, api |
| Driven adapters (DB) | `backend/app/adapters/driven/database/*.py` | repository, db, query, persistence |
| Driven adapters (External) | `backend/app/adapters/driven/{stripe,google,clerk,email,twilio,posthog}/*.py` | client, external, integration |
| Ports (interfaces) | `backend/app/ports/*.py` | port, interface, abstract, contract |
| Domain entities | `backend/app/domain/*/entities.py` | entity, domain, model, business |
| Domain exceptions | `backend/app/domain/*/exceptions.py` | exception, error, domain |
| Services | `backend/app/services/*.py` | service, business, logic |

---

## SQLAlchemy Models

All in `backend/app/adapters/driven/database/models.py`. Account is the multi-tenancy root — all tenant data hangs off `account_id`.

| Model | Key Fields | Relationships |
|-------|-----------|---------------|
| **Account** | subscription_status (trial/active/past_due/canceled/paused), stripe_customer_id, trial_ends_at, customer_count | users (1:N), customers (1:N) |
| **User** | account_id (FK), email, clerk_id (unique), role (owner/admin/member), is_active | account (N:1), credentials (1:1) |
| **Customer** | account_id (FK), name, phone, email, contract_status, google_contact_id, source, deleted_at | timeline_events (1:N), tasks (1:N), notes (1:N) |
| **TimelineEvent** | account_id (FK), customer_id (FK), type (service_completed/call_attempted/call_completed/reminder_dismissed/note_added/contract_signed), date, value, event_data (JSON) | customer (N:1) |
| **UserTask** | account_id (FK), customer_id (FK, nullable), title, importance (red/yellow/green), due_date, completed, google_event_id | — |
| **UserCredentials** | user_id (FK, unique), google_access_token (encrypted), google_refresh_token (encrypted), calendar_connected | — |
| **Note** | account_id (FK), customer_id (FK), user_id (FK), content, note_type (general/call_note/meeting_note), is_pinned, deleted_at | — |
| **SyncLog** | account_id (FK), sync_type, status, records_processed/created/updated/skipped/failed, sync_metadata (JSON) | — |
| **AuditLog** | account_id (FK), user_id (FK), action, resource_type, resource_id, ip_address, extra_data (JSON) | — |
| **AnalyticsEvent** | account_id (FK), user_id (FK), event_type, event_metadata (JSON) | — |
| **RefreshToken** | token_hash, user_id (FK), expires_at, revoked, device_info, ip_address | — |

---

## Dependency Graph

```
main.py
├── database (get_db, init_db)
├── models (all ORM models)
├── config (settings)
├── clerk_auth (get_current_user, get_current_account)
├── clerk_webhook (router)
├── security (middleware, validators, subscription checks)
├── encryption (encrypt_token, decrypt_token)
├── scoring (calculate_customer_metrics, calculate_priority_score)
├── recommendations (get_recommendation, get_color_bucket, get_emoji)
├── google_calendar (OAuth + API calls)
├── billing (Stripe operations)
├── email_service (Resend API)
├── demo_data (generators)
└── logging_config (structured logging)
```

No circular dependencies. `main.py` is a leaf consumer — nothing imports from it except tests.

---

## Workflow & Automation

For the full workflow system documentation — the Ralph loop, scaffolding, audit pipeline, and all tooling — see `agents/README.md`.

---

## Security Findings (02/14/26)

Non-PASS items only, sorted by severity. Full audit in `agents/archive/security/audit-02-14-26.md`.

### CRITICAL

| ID | Finding | Files |
|----|---------|-------|
| 04-1 | **OAuth tokens stored in plaintext** — `encrypt_token()`/`decrypt_token()` exist but are dead code | repositories/calendar.py:54-55,77,84,90 |

### HIGH

| ID | Finding | Files |
|----|---------|-------|
| 04-2 | OAuth state unsigned — nonce generated but never verified | google/client.py:101-125, services/calendar.py:60-73 |
| 04-3 | OAuth callback has no authentication — attacker can link Google to any user | calendar.py:63-80, services/calendar.py:52-94 |
| 05-3 | Customer limit missing default clause (dead code) | security.py:278-323 |
| 05-5 | Customer limit pre-import check not wired — trial users can import unlimited | contacts.py:143-147, services/contacts.py:449,492,614,627 |
| 11-1 | Nginx CSP has unsafe-eval — nullifies XSS protection | nginx.conf:15 |
| 11-3 | Nginx header inheritance drops all security headers on SPA pages | nginx.conf:11-15, 42-46, 62-68 |

### MEDIUM

| ID | Finding | Files |
|----|---------|-------|
| 01-2b | azp (authorized party) not validated in JWT | clerk_auth.py:85-94 |
| 01-4b | Race condition on email linking (check-then-act, no lock) | clerk_auth.py:148-161 |
| 01-5 | Inactive user bypass via Clerk re-registration | clerk_auth.py:163-195 |
| 04-4 | Encryption empty string bypass | encryption.py:37-38, 53-54 |
| 05-2 | Notes routes missing subscription gates | notes.py:40, 96, 126 |
| 06-2 | Rate limit key uses proxy IP — all users share one bucket | security.py:14,20; main.py:78 |
| 11-2 | Nginx CSP missing PostHog and Clerk domains | nginx.conf:15 |
| 12-1 | Startup validation warnings don't block production | main.py:119, config.py:264 |
| 12-6 | Dependency vulnerabilities (python-multipart, cryptography, passlib) | requirements.txt |

### LOW

| ID | Finding |
|----|---------|
| 01-3a | JWKS domain not validated |
| 01-4c | Unlimited account creation (no rate limit on Clerk signup) |
| 03a-2 | Open redirect — javascript: scheme bypass (mitigated by Stripe) |
| 03a-3 | CSV import call_outcome not sanitized |
| 03b-1 | Timeline event leak — raw content preview |
| 03b-2 | Estimate description missing backend sanitization (React escapes) |
| 03b-3 | Action notes missing backend sanitization (React escapes) |
| 04-5 | OAuth key validation is warning not error |
| 04-7 | Token exposure possible in tracebacks |
| 06-1 | Dead rate limit constants |
| 06-5 | Import endpoint synchronous CSV parsing |
| 06-6 | Unprotected root endpoint |
| 07-1 | Domain exceptions expose internal IDs |
| 07-2 | Clerk webhook leaks internal IDs (signature-verified) |
| 07-3 | Clerk webhook confirms user existence (signature-verified) |
| 08-2 | localStorage autosave includes customer PII |
| 08-4 | API base URL fallback to localhost |
| 08-6 | PostHog autocapture PII surface |
| 09-2 | Clerk webhook replay could revert user data (5-min window) |
| 09-4 | Clerk webhook race condition on user creation |
| 09-6 | Event type reflection in webhook response |
| 10-3 | Stripe integer parsing crash on bad metadata |
| 10-4 | Stripe status map passthrough to DB |
| 10-5 | Stripe event handler idempotency gap |
| 11-6 | Missing Cache-Control on API responses |
| 12-2 | Localhost bypass in URL validation (127.0.0.1, ::1) |
| 12-3 | Secret key no entropy check |
| 12-4 | Encryption key validation is warning not error |
| 12-5 | Environment detection no enum constraint |
