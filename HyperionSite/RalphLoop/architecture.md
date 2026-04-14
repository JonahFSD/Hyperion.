---
tags:
  - architecture
  - hexagonal
  - backend
  - fastapi
  - services
  - ports
  - adapters
  - dependency-injection
status: current
---

# Backend Architecture Map

> Generated for hexagonal architecture refactor planning.
> Source: `backend/main.py` (2,515 lines), `backend/models.py` (308 lines), plus 18 supporting modules.

---

## File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 2,515 | Monolithic FastAPI app (all endpoints + schemas + helpers) |
| `models.py` | 308 | All SQLAlchemy models |
| `database.py` | 60 | DB engine, session factory, `get_db` |
| `config.py` | 306 | Settings via pydantic-settings |
| `clerk_auth.py` | 145 | Clerk JWT validation, `get_current_user`/`get_current_account` |
| `clerk_webhook.py` | 224 | Clerk user lifecycle webhooks |
| `security.py` | 360 | Rate limiting, sanitization, security headers, subscription checks |
| `security_hardening.py` | 592 | Advanced security middleware |
| `audit.py` | 345 | Audit log helpers |
| `encryption.py` | 84 | Fernet encrypt/decrypt for OAuth tokens |
| `customer_helpers.py` | 138 | Batch metric fetching (N+1 prevention) |
| `scoring.py` | 220 | Priority scoring algorithm (0-100) |
| `recommendations.py` | 280 | Action recommendation engine |
| `subscription.py` | 87 | Subscription tier enforcement dependencies |
| `google_calendar.py` | 513 | Google OAuth + Calendar + Contacts API |
| `google_auth_helper.py` | 303 | Google auth utilities |
| `stripe_integration.py` | 325 | Stripe API client |
| `billing.py` | 382 | Billing business logic |
| `email_service.py` | 554 | Resend email integration |
| `email_notifications.py` | 333 | Email templates |
| `logging_config.py` | 443 | Structured logging |
| `demo_data.py` | 264 | Demo customer generator |
| `seed_data.py` | 367 | Database seeding |

**Total backend Python:** ~8,698 lines

---

## SQLAlchemy Models

### Account
- **Multi-tenancy root.** All tenant data hangs off `account_id`.
- Fields: `id`, `name`, `subscription_status`, `subscription_id`, `stripe_customer_id`, `trial_ends_at`, `current_period_end`, `customer_count`, `created_at`, `updated_at`
- Relationships: `users` (1:N), `customers` (1:N)
- Constraints: `subscription_status` CHECK (trial, active, past_due, canceled, paused)

### User
- Fields: `id`, `account_id` (FK), `email`, `hashed_password`, `full_name`, `role`, `is_active`, `must_change_password`, `clerk_id` (unique), `email_verified`, `email_verification_token`, `email_verification_sent_at`, `password_reset_token`, `password_reset_expiry`, `failed_login_attempts`, `locked_until`, `created_at`, `updated_at`, `last_login`
- Relationships: `account` (N:1), `credentials` (1:1), `refresh_tokens` (1:N)
- Constraints: unique(`account_id`, `email`), `role` CHECK (owner, admin, member)
- Indexes: `account_id`, `email`, `clerk_id`

### Customer
- Fields: `id`, `account_id` (FK), `name`, `phone`, `email`, `contact_preference`, `contract_status`, `contract_expiry`, `google_contact_id`, `source`, `last_synced_at`, `created_at`, `updated_at`, `deleted_at`
- Relationships: `account` (N:1), `timeline_events` (1:N), `tasks` (1:N), `notes` (1:N)
- Constraints: `contract_status` CHECK, `source` CHECK
- Indexes: `account_id`, (`account_id`, `phone`), (`account_id`, `google_contact_id`)

### TimelineEvent
- Fields: `id`, `account_id` (FK), `customer_id` (FK), `type`, `date`, `value`, `event_data` (JSON), `updated_at`
- Relationships: `customer` (N:1)
- Constraints: `type` CHECK (service_completed, call_attempted, call_completed, reminder_dismissed, note_added, contract_signed)
- account_id is denormalized for query performance

### UserTask
- Fields: `id`, `account_id` (FK), `customer_id` (FK, nullable), `title`, `importance`, `due_date`, `completed`, `google_event_id`, `created_at`, `updated_at`
- Constraints: `importance` CHECK (red, yellow, green)

### UserCredentials
- Fields: `id`, `user_id` (FK, unique), `google_access_token` (encrypted), `google_refresh_token` (encrypted), `google_token_expiry`, `calendar_connected`, `created_at`, `updated_at`
- Linked to account via `user.account_id`

### AnalyticsEvent
- Fields: `id`, `account_id` (FK), `user_id` (FK), `event_type`, `timestamp`, `event_metadata` (JSON)
- Indexes: `event_type`, `timestamp`, (`account_id`, `event_type`)

### RefreshToken
- Fields: `id`, `token_hash`, `user_id` (FK), `expires_at`, `revoked`, `revoked_at`, `device_info`, `ip_address`, `created_at`
- Linked to account via `user.account_id`

### SyncLog
- Fields: `id`, `account_id` (FK), `user_id` (FK), `sync_type`, `status`, `started_at`, `completed_at`, `records_processed`, `records_created`, `records_updated`, `records_skipped`, `records_failed`, `error_message`, `sync_metadata` (JSON), `created_at`
- Constraints: `status` CHECK, `sync_type` CHECK
- Indexes: (`account_id`, `sync_type`, `created_at`)

### Note
- Fields: `id`, `account_id` (FK), `customer_id` (FK), `user_id` (FK), `content`, `note_type`, `is_pinned`, `created_at`, `updated_at`, `deleted_at`
- Constraints: `note_type` CHECK (general, call_note, meeting_note)
- Indexes: (`customer_id`, `is_pinned`, `created_at`)

### AuditLog
- Fields: `id`, `account_id` (FK), `user_id` (FK), `action`, `resource_type`, `resource_id`, `ip_address`, `user_agent`, `extra_data` (JSON), `timestamp`
- Indexes: (`user_id`, `action`, `timestamp`), (`account_id`, `action`, `timestamp`), (`resource_type`, `resource_id`, `timestamp`)

---

## Pydantic Schemas (all in main.py)

| Schema | Lines | Fields |
|--------|-------|--------|
| `ActionRequest` | 124-131 | customer_id, action_type, outcome, scheduled_date, dismiss_reason, notes, job_value |
| `CustomerCard` | 133-149 | id, name, phone, email, score, color, emoji, priority, avg_invoice, days_since_service, recommendation, contract_status, contract_expiry |
| `CustomerDetail` | 151-166 | Extends CustomerCard + timeline, score_breakdown |
| `NoteCreate` | 1519-1521 | content, note_type |
| `NoteUpdate` | 1524-1526 | content (opt), is_pinned (opt) |
| `CheckoutRequest` | 1867-1869 | success_url (opt), cancel_url (opt) |
| `TeamMemberResponse` | 2105-2116 | id, email, full_name, role, is_active, created_at, last_login |
| `TeamMemberInvite` | 2119-2122 | email, role |
| `TeamMemberRoleUpdate` | 2125-2127 | role |

---

## Domain: Contacts (Customers)

### Endpoints

| # | Method | Path | Auth | Models | Lines |
|---|--------|------|------|--------|-------|
| 1 | GET | `/api/today` | require_subscription | Customer, UserTask | 496-585 |
| 2 | GET | `/api/customer/{customer_id}` | require_subscription | Customer, TimelineEvent | 587-655 |
| 3 | POST | `/api/action` | require_active_subscription | Customer, TimelineEvent, UserTask, UserCredentials | 657-800 |
| 4 | POST | `/api/import` | require_active_subscription | Customer, TimelineEvent, UserTask | 802-1008 |
| 5 | GET | `/api/import/template` | Public | None | 1010-1022 |
| 6 | POST | `/api/demo-customers` | User + Account | Customer, TimelineEvent, AnalyticsEvent | 354-423 |
| 7 | DELETE | `/api/customers/clear-all` | User + Account (owner/admin) | Customer, TimelineEvent, Note, UserTask, AnalyticsEvent | 426-489 |

### Helpers (business logic → service layer)

| Function | Location | Purpose | Called By |
|----------|----------|---------|-----------|
| `calculate_customer_metrics` | scoring.py | Compute days_since, avg_invoice, etc. | today endpoint |
| `calculate_priority_score` | scoring.py | 0-100 priority score | today endpoint |
| `get_recommendation` | recommendations.py | Action suggestion | today endpoint |
| `get_color_bucket` | recommendations.py | Score → red/yellow/green | today endpoint |
| `get_emoji` | recommendations.py | Color → emoji | today endpoint |
| `fetch_all_customer_metrics` | customer_helpers.py | Batch metric fetch (N+1 fix) | today endpoint |
| `calculate_customer_scores_and_recommendations` | customer_helpers.py | Batch score+recommend | today endpoint |
| `get_customer_metrics_and_score` | customer_helpers.py | Single customer score | detail endpoint |
| `sanitize_csv_value` | security.py | CSV injection prevention | import endpoint |
| `validate_file_upload` | security.py | File validation | import endpoint |
| `sanitize_phone_number` | security.py | Phone normalization | import endpoint |

### Notes Sub-domain

| # | Method | Path | Auth | Models | Lines |
|---|--------|------|------|--------|-------|
| 1 | GET | `/api/customer/{customer_id}/notes` | User | Customer, Note | 1529-1573 |
| 2 | POST | `/api/customer/{customer_id}/notes` | require_active_subscription | Customer, Note, TimelineEvent | 1576-1631 |
| 3 | PATCH | `/api/notes/{note_id}` | User | Note | 1634-1673 |
| 4 | DELETE | `/api/notes/{note_id}` | User | Note | 1676-1698 |

### Estimated extraction size: ~700 lines endpoints + ~638 lines helpers = ~1,338 lines

---

## Domain: Calendar (Google Integration)

### Endpoints

| # | Method | Path | Auth | Models | Lines |
|---|--------|------|------|--------|-------|
| 1 | GET | `/api/auth/google/connect` | User | User | 1028-1040 |
| 2 | GET | `/api/auth/google/callback` | Public (OAuth) | User, UserCredentials | 1042-1112 |
| 3 | GET | `/api/auth/google/status` | User | UserCredentials | 1114-1127 |
| 4 | POST | `/api/auth/google/disconnect` | User | UserCredentials | 1129-1146 |
| 5 | GET | `/api/calendar/events` | User | UserCredentials | 1148-1197 |
| 6 | GET | `/api/google/contacts-preview` | User | Customer, UserCredentials | 1199-1290 |
| 7 | POST | `/api/google/import-contacts` | User + Account | Customer, UserCredentials, SyncLog, AnalyticsEvent | 1292-1513 |

### Helpers (infrastructure → adapter layer)

| Function | Location | Purpose |
|----------|----------|---------|
| `get_authorization_url` | google_calendar.py | Build OAuth URL |
| `exchange_code_for_tokens` | google_calendar.py | OAuth code → tokens |
| `decode_oauth_state` | google_calendar.py | Validate state param |
| `refresh_access_token` | google_calendar.py | Token refresh |
| `create_calendar_event` | google_calendar.py | Create GCal event |
| `list_upcoming_events` | google_calendar.py | Fetch events |
| `fetch_google_contacts` | google_calendar.py | People API fetch |
| `encrypt_token` / `decrypt_token` | encryption.py | Token at-rest encryption |

### Estimated extraction size: ~490 lines endpoints + ~816 lines helpers = ~1,306 lines

---

## Domain: Billing (Stripe)

### Endpoints

| # | Method | Path | Auth | Models | Lines |
|---|--------|------|------|--------|-------|
| 1 | POST | `/api/billing/checkout` | User + Account | Account | 1872-1904 |
| 2 | POST | `/api/billing/portal` | User + Account | Account | 1907-1927 |
| 3 | GET | `/api/billing/subscription` | User + Account | Account | 1930-1959 |
| 4 | POST | `/api/billing/cancel` | User + Account | Account, AuditLog | 1962-1990 |
| 5 | POST | `/api/billing/reactivate` | User + Account | Account, AuditLog | 1993-2019 |
| 6 | POST | `/api/billing/upgrade` | User + Account | Account, AuditLog | 2022-2079 |
| 7 | POST | `/api/billing/webhook` | Public (Stripe sig) | None | 2082-2095 |

### Helpers (infrastructure → adapter layer)

| Function | Location | Purpose |
|----------|----------|---------|
| `is_stripe_configured` | billing.py | Check Stripe keys |
| `create_checkout_session` | billing.py | Create Stripe checkout |
| `create_customer_portal_session` | billing.py | Portal session |
| `get_subscription_details` | billing.py | Fetch sub from Stripe |
| `cancel_subscription` | billing.py | Cancel via Stripe |
| `reactivate_subscription` | billing.py | Reactivate via Stripe |
| `process_webhook_event` | billing.py | Webhook event handler |
| `check_subscription_status` | security.py | Enforce tier limits |
| `check_customer_limit` | security.py | Customer count enforcement |
| `get_customer_limit_for_account` | security.py | Tier → limit mapping |
| `require_subscription` | subscription.py | Dependency: read access |
| `require_active_subscription` | subscription.py | Dependency: write access |

### Estimated extraction size: ~230 lines endpoints + ~794 lines helpers = ~1,024 lines

---

## Domain: Team Management

### Endpoints

| # | Method | Path | Auth | Models | Lines |
|---|--------|------|------|--------|-------|
| 1 | GET | `/api/team` | User + Account | User | 2205-2236 |
| 2 | POST | `/api/team/invite` | User + Account (owner/admin) | User, AnalyticsEvent, AuditLog | 2239-2344 |
| 3 | PUT | `/api/team/{user_id}/role` | User + Account (owner/admin) | User, AuditLog | 2347-2431 |
| 4 | DELETE | `/api/team/{user_id}` | User + Account (owner/admin) | User, AuditLog | 2434-2510 |

### Helpers

| Function | Location | Purpose |
|----------|----------|---------|
| `send_team_invite_email` | email_service.py | Send invite via Resend |

### Estimated extraction size: ~310 lines endpoints + ~554 lines email = ~864 lines

---

## Domain: Auth (Clerk)

### Endpoints

| # | Method | Path | Auth | Models | Lines |
|---|--------|------|------|--------|-------|
| 1 | GET | `/api/auth/me` | User + Account | User, Account | 326-351 |
| 2 | POST | `/api/webhooks/clerk` | Clerk sig | User, Account | (clerk_webhook.py) |

### Helpers (infrastructure → adapter layer)

| Function | Location | Purpose |
|----------|----------|---------|
| `get_current_user` | clerk_auth.py | JWT → User lookup |
| `get_current_account` | clerk_auth.py | User → Account lookup |
| Clerk webhook handlers | clerk_webhook.py | user.created, user.updated, user.deleted |

### Estimated extraction size: ~30 lines endpoint + ~369 lines helpers = ~399 lines

---

## Domain: Admin (Audit, Health, Sync)

### Endpoints

| # | Method | Path | Auth | Models | Lines |
|---|--------|------|------|--------|-------|
| 1 | GET | `/` | Public | None | 206-208 |
| 2 | GET | `/api/health` | Public | None | 217-225 |
| 3 | GET | `/api/health/db` | User (owner/admin) | User | 228-261 |
| 4 | GET | `/api/health/full` | User (owner/admin) | User | 264-319 |
| 5 | GET | `/api/audit/logs` | User (owner/admin) | AuditLog | 1793-1848 |
| 6 | GET | `/api/sync/history` | User | SyncLog | 1705-1750 |
| 7 | GET | `/api/sync/{sync_id}` | User | SyncLog | 1753-1786 |

### Helpers

| Function | Location | Purpose |
|----------|----------|---------|
| `audit_log` | audit.py | Write audit entry |
| `escape_like` | main.py:18-25 | SQL LIKE injection prevention |

### KPI Metrics

| # | Method | Path | Auth | Models | Lines |
|---|--------|------|------|--------|-------|
| 1 | GET | `/api/metrics/kpi` | User + Account | TimelineEvent | 2134-2202 |

### Estimated extraction size: ~260 lines endpoints + ~345 lines audit = ~605 lines

---

## Cross-Cutting Concerns (Middleware / Infra)

| Concern | Location | Lines | Notes |
|---------|----------|-------|-------|
| Request ID middleware | main.py:106-120 | 15 | Adds X-Request-ID |
| Global exception handler | main.py:183-203 | 20 | Error sanitization |
| CORS config | main.py:85-97 | 12 | Origin whitelist |
| Rate limiting | security.py | ~60 | slowapi-based |
| Security headers | security.py | ~40 | HSTS, CSP, etc. |
| Advanced hardening | security_hardening.py | 592 | Additional protections |
| Structured logging | logging_config.py | 443 | Request-scoped logs |
| Sentry integration | main.py:72-80 | 8 | Error monitoring |

---

## Dependency Graph

```
main.py
├── database.py (get_db, init_db)
├── models.py (all 11 models)
├── config.py (settings)
├── clerk_auth.py (get_current_user, get_current_account)
│   └── models.py, database.py, config.py
├── clerk_webhook.py (router)
│   └── models.py, database.py, config.py
├── security.py (middleware, validators, subscription checks)
│   └── config.py, models.py
├── subscription.py (require_subscription, require_active_subscription)
│   └── clerk_auth.py, security.py
├── encryption.py (encrypt_token, decrypt_token)
│   └── config.py
├── audit.py (audit_log, AuditAction)
│   └── models.py, database.py
├── scoring.py (calculate_customer_metrics, calculate_priority_score)
│   └── (pure functions, no imports from project)
├── recommendations.py (get_recommendation, get_color_bucket, get_emoji)
│   └── (pure functions, no imports from project)
├── customer_helpers.py (batch fetchers)
│   └── models.py, scoring.py, recommendations.py
├── google_calendar.py (OAuth + API calls)
│   └── config.py
├── billing.py (Stripe operations)
│   └── config.py, stripe_integration.py
├── email_service.py (Resend API)
│   └── config.py
├── demo_data.py (generators)
│   └── (no project imports)
└── logging_config.py (structured logging)
    └── config.py
```

**No circular dependencies detected.** `main.py` is a leaf consumer — nothing imports from it except tests.

---

## Summary: Extraction Priorities

| Domain | Endpoint Lines | Helper Lines | Total | Complexity |
|--------|---------------|--------------|-------|------------|
| Contacts | ~700 | ~638 | ~1,338 | High (scoring, import, actions) |
| Calendar | ~490 | ~816 | ~1,306 | High (OAuth, contacts sync) |
| Billing | ~230 | ~794 | ~1,024 | Medium (Stripe adapter exists) |
| Team | ~310 | ~554 | ~864 | Medium (email, RBAC) |
| Admin | ~260 | ~345 | ~605 | Low (mostly queries) |
| Auth | ~30 | ~369 | ~399 | Low (Clerk does heavy lifting) |

**main.py breakdown:**
- Endpoints: ~2,020 lines (80%)
- Schemas: ~80 lines (3%)
- Middleware/setup: ~170 lines (7%)
- Helpers: ~245 lines (10%)
