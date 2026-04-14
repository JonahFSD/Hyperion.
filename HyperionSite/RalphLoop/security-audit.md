# Reusable Security Audit Prompt

> Run this in Claude Code CLI periodically (quarterly recommended).
> Start by reading `specs/README.md` for the file index — do NOT grep blindly.

---

```
## Instructions

You are performing a security audit of the CallSheet CRM application. This is a multi-tenant SaaS app with:
- **Frontend:** React SPA with Clerk authentication
- **Backend:** Python FastAPI with hexagonal architecture
- **Auth:** Clerk JWT (RS256 via JWKS), webhook-based user provisioning
- **Payments:** Stripe with webhook signature verification
- **OAuth:** Google Calendar/Contacts via OAuth 2.0, tokens encrypted with Fernet
- **SMS:** Twilio for estimate delivery
- **Email:** Resend for transactional email
- **Database:** PostgreSQL (prod) / SQLite (dev) via SQLAlchemy ORM

**IMPORTANT:** Before grepping the codebase, read `specs/README.md` first. It contains a complete file index organized by domain with exact paths. Use it.

For each category below, read the specific files listed, evaluate against the criteria, and rate as:
- 🔴 **CRITICAL** — exploitable vulnerability, fix immediately
- 🟠 **HIGH** — significant risk, fix within a sprint
- 🟡 **MEDIUM** — defense-in-depth gap, fix when convenient
- ✅ **PASS** — no issues found

Output the full report to `specs/security/audit-YYYY-MM-DD.md` (create the directory if needed).

---

## CATEGORY 1: Authentication & Session Management

**Files to read:**
- `backend/clerk_auth.py` — JWT validation, user resolution, fallback user creation
- `backend/clerk_webhook.py` — user lifecycle webhook handling
- `backend/app/adapters/driving/api/dependencies.py` — auth dependency injection

**Check:**
- [ ] JWT verification uses RS256 with JWKS key rotation (not a static secret)
- [ ] Token expiration (`exp`) is verified and enforced
- [ ] Required claims (`sub`, `exp`, `iat`) are validated
- [ ] Fallback user creation (when webhook fails) cannot be exploited to create unauthorized accounts
- [ ] Webhook signature verification uses constant-time comparison (`hmac.compare_digest`)
- [ ] Webhook timestamp is checked (replay attack prevention) — verify the tolerance window is ≤5 minutes
- [ ] Deactivated users (`is_active=False`) are blocked from all endpoints
- [ ] The `get_current_user` dependency is applied to EVERY route that requires auth — check each route file
- [ ] No route handler accepts a user_id from the request body when it should use the authenticated user

**Known implementation details:**
- Clerk JWTs are verified against JWKS endpoint derived from `CLERK_PUBLISHABLE_KEY`
- Webhook uses Svix protocol (HMAC-SHA256) with `whsec_` prefixed secrets
- Fallback user creation calls Clerk API to verify the user exists before creating locally

---

## CATEGORY 2: Authorization & Tenant Isolation (IDOR)

**Files to read:**
- `backend/app/adapters/driving/api/contacts.py` — customer CRUD, notes, import
- `backend/app/adapters/driving/api/calendar.py` — calendar events, Google OAuth
- `backend/app/adapters/driving/api/billing.py` — subscription management
- `backend/app/adapters/driving/api/team.py` — team member management
- `backend/app/adapters/driving/api/estimates.py` — estimate CRUD
- `backend/app/services/contacts.py` — business logic for contacts

**Check:**
- [ ] Every database query that fetches a resource by ID also filters by `account_id` (tenant isolation)
- [ ] `GET /customer/{id}` cannot return a customer belonging to a different account
- [ ] `POST /action` verifies the customer belongs to the authenticated user's account
- [ ] `DELETE /customers/clear-all` is scoped to the authenticated user's account only
- [ ] `GET /customer/{id}/notes` and `POST /customer/{id}/notes` verify customer ownership
- [ ] `PATCH /notes/{id}` and `DELETE /notes/{id}` verify the note belongs to the user's account
- [ ] Calendar credentials are scoped to account — one account cannot read another's Google tokens
- [ ] Team endpoints verify the requesting user has appropriate role (owner/admin) for management actions
- [ ] `PUT /{user_id}/role` cannot be used to escalate privileges (e.g., member promoting self to owner)
- [ ] Estimate endpoints verify ownership — one account cannot view/modify another's estimates
- [ ] The demo endpoint (`/demo-customers`) is properly isolated per account
- [ ] No endpoint accepts `account_id` as a request parameter — it should ALWAYS come from the auth context

**IDOR test pattern:** For each endpoint that takes a resource ID in the URL path, trace the code path and verify the query includes `WHERE account_id = <authenticated_account_id>`.

---

## CATEGORY 3: Input Validation & Injection

**Files to read:**
- `backend/app/adapters/driving/api/schemas.py` — Pydantic request/response models
- `backend/security.py` — sanitization utilities
- `backend/app/services/contacts.py` — import processing logic

**Check:**
- [ ] Every POST/PUT/PATCH endpoint validates input through Pydantic schemas
- [ ] String fields have `max_length` constraints in Pydantic models
- [ ] The CSV import endpoint (`POST /import`) validates:
  - File size (current limit: 5MB via `validate_file_upload`)
  - File extension whitelist (only `.csv`)
  - MIME type validation
  - Row count limit (prevent importing millions of rows)
  - Cell content sanitization (CSV injection via `sanitize_csv_value`)
- [ ] Phone number input is sanitized (`sanitize_phone_number` — digits, +, -, spaces, parens only)
- [ ] Email format is validated before storage
- [ ] HTML content is escaped (`sanitize_html`) before storage — verify this is called on user-provided text fields (notes, names, etc.)
- [ ] No raw SQL queries exist anywhere — all DB access goes through SQLAlchemy ORM
- [ ] Note content and customer names cannot contain executable HTML/JS that would render in the frontend
- [ ] File upload filenames are sanitized (`sanitize_filename`) before any disk operation
- [ ] The import template download (`GET /import/template`) serves a static file, not user-controlled content

**SQL injection specific:** Search for any use of `text()`, `execute()`, `raw()`, or string formatting in SQL queries:
```bash
grep -rn "\.execute\|\.text(\|f\".*SELECT\|f\".*INSERT\|f\".*UPDATE\|f\".*DELETE" backend/app/
```

---

## CATEGORY 4: Secrets & Configuration

**Files to read:**
- `backend/config.py` — environment variable loading and validation
- `backend/encryption.py` — Fernet encryption for OAuth tokens
- `backend/.env` (if exists) — check for committed secrets
- `.gitignore` — verify secret files are excluded

**Check:**
- [ ] `SECRET_KEY` has no default value (must be set via environment)
- [ ] `ENCRYPTION_KEY` is validated for correct Fernet format (44-char base64)
- [ ] Production config validation rejects: default SECRET_KEY, SQLite database, localhost URLs, missing Stripe keys
- [ ] `.env` file is in `.gitignore`
- [ ] No secrets appear in committed code — search for patterns:
  ```bash
  grep -rn "sk_live_\|sk_test_\|whsec_\|re_\|CHANGE-THIS" backend/ frontend/ --include="*.py" --include="*.js" --include="*.jsx" --include="*.json"
  ```
- [ ] Frontend bundle contains ONLY the Clerk publishable key (starts with `pk_`) — no secret keys
- [ ] Google OAuth `client_secret` is never exposed to the frontend
- [ ] Stripe secret key is never exposed to the frontend
- [ ] Twilio auth token is never exposed to the frontend
- [ ] The `encryption_key` config field uses `Optional[str]` — verify it's never logged or included in error responses
- [ ] Config validation runs on startup (`validate_on_startup` called from `main.py`)

**Frontend secrets check:**
```bash
grep -rn "sk_\|secret\|password\|auth_token\|api_key" frontend/src/ --include="*.js" --include="*.jsx" | grep -v "node_modules" | grep -v ".css"
```

---

## CATEGORY 5: API Security & Rate Limiting

**Files to read:**
- `backend/security.py` — rate limiter, security headers middleware
- `backend/main.py` — middleware registration
- All route files in `backend/app/adapters/driving/api/`

**Check:**
- [ ] `SecurityHeadersMiddleware` is registered in `main.py`
- [ ] Security headers include: X-Content-Type-Options, X-Frame-Options, HSTS, CSP, Referrer-Policy, Permissions-Policy
- [ ] CSP policy restricts script-src to 'self' and necessary CDNs (Stripe)
- [ ] Rate limiting is applied to sensitive endpoints:
  - Auth/login: 10/15min (`RATE_LIMIT_AUTH`)
  - General API: 100/min (`RATE_LIMIT_API`)
  - File upload: 10/hour (`RATE_LIMIT_UPLOAD`)
  - Webhook: has its own limit
- [ ] Rate limiting uses IP-based key (`get_remote_address`) — verify it handles X-Forwarded-For correctly behind a proxy
- [ ] CORS is configured to allow only the frontend domain, not `*`
- [ ] The `/health` endpoint does not require auth (intentional) but does not expose sensitive info
- [ ] Error responses do not leak stack traces, SQL queries, or internal file paths in production
- [ ] The Stripe webhook endpoint (`POST /webhook`) verifies the Stripe signature before processing
- [ ] All DELETE endpoints require authentication AND authorization

---

## CATEGORY 6: OAuth Token Security

**Files to read:**
- `backend/encryption.py` — Fernet encryption service
- `backend/app/adapters/driven/google/client.py` — Google OAuth client
- `backend/app/services/calendar.py` — calendar service (token usage)
- `backend/app/adapters/driven/database/repositories/calendar.py` — credentials storage

**Check:**
- [ ] Google OAuth access tokens and refresh tokens are encrypted at rest using Fernet
- [ ] Tokens are decrypted only at the point of use (not stored decrypted in memory longer than needed)
- [ ] Token refresh logic handles expired/revoked tokens gracefully without exposing the refresh token
- [ ] OAuth callback validates the `state` parameter to prevent CSRF
- [ ] The Google redirect URI is configured per environment (not hardcoded to localhost in production)
- [ ] Disconnecting Google (`POST /auth/google/disconnect`) deletes stored credentials, not just deactivates
- [ ] Google API scopes are minimal (only calendar and contacts, not full account access)
- [ ] Failed decryption (corrupted/rotated key) is handled gracefully — check the `except` in `decrypt()`

---

## CATEGORY 7: Payment & Billing Security

**Files to read:**
- `backend/app/adapters/driving/api/billing.py` — billing routes
- `backend/app/services/billing.py` — billing business logic
- `backend/app/adapters/driven/stripe/client.py` — Stripe integration
- `backend/security.py` — subscription status checks

**Check:**
- [ ] Stripe webhook signature is verified before processing any event
- [ ] Subscription status changes only happen through Stripe webhooks, not client-side requests
- [ ] `check_subscription_status()` cannot be bypassed by manipulating request headers
- [ ] Customer limit enforcement (`check_customer_limit`) runs BEFORE bulk import, not after
- [ ] The billing portal URL is generated server-side (Stripe creates it, not the client)
- [ ] Checkout session creation validates the price_id against known/allowed values
- [ ] Subscription tier cannot be spoofed — the tier comes from Stripe, not the client
- [ ] Past-due accounts have appropriately restricted access (read-only, per current implementation)
- [ ] Trial expiration is enforced server-side, not just client-side UI

---

## CATEGORY 8: Data Exposure & Privacy

**Files to read:**
- `backend/app/adapters/driving/api/schemas.py` — response models
- `backend/app/adapters/driving/api/demo.py` — demo data endpoint
- Frontend `api.js` — what data the frontend requests

**Check:**
- [ ] API responses use Pydantic response models (not raw database objects that might leak fields)
- [ ] User passwords/hashes are never included in API responses
- [ ] OAuth tokens (encrypted or plaintext) are never included in API responses
- [ ] Internal IDs (account_id) are not leaked in responses where they shouldn't be
- [ ] Error messages don't reveal whether a specific email/phone exists in the system (enumeration prevention)
- [ ] The demo endpoint doesn't expose real customer data from other accounts
- [ ] Audit logs (`/audit/logs`) are restricted to admin/owner roles
- [ ] KPI metrics (`/metrics/kpi`) are scoped to the requesting account

---

## CATEGORY 9: Frontend Security

**Files to read:**
- `frontend/src/api.js` — API client, token handling
- `frontend/src/contexts/ClerkAuthBridge.jsx` — auth context
- `frontend/src/App.jsx` — route guards
- `frontend/src/components/SubscriptionGate.jsx` — paywall enforcement

**Check:**
- [ ] Auth tokens are stored via Clerk's built-in mechanism (not manually in localStorage/cookies)
- [ ] API requests include Bearer token via the centralized api.js module
- [ ] No API calls bypass the centralized module (no direct fetch() in components)
- [ ] Subscription gating is enforced server-side (not just hidden UI elements)
- [ ] Sensitive routes (settings, billing) are behind auth checks, not just hidden navigation
- [ ] No `dangerouslySetInnerHTML` usage with user-provided content
- [ ] React's built-in XSS protection (JSX escaping) is not bypassed anywhere
- [ ] The Clerk publishable key is the ONLY credential in the frontend bundle
- [ ] Source maps are disabled in production builds

**Check for dangerous patterns:**
```bash
grep -rn "dangerouslySetInnerHTML\|eval(\|innerHTML\|document.write" frontend/src/ --include="*.jsx" --include="*.js"
```

---

## CATEGORY 10: Infrastructure & Deployment

**Files to read:**
- `backend/main.py` — app initialization, middleware stack
- `backend/logging_config.py` — logging configuration
- `backend/database.py` or equivalent — database connection setup
- Any Dockerfile, docker-compose.yml, or deployment configs

**Check:**
- [ ] Debug mode is disabled in production
- [ ] Logging does not include sensitive data (tokens, passwords, API keys)
- [ ] Database connections use connection pooling with reasonable limits (current: pool_size=20, max_overflow=10)
- [ ] Database connection string uses SSL in production
- [ ] CORS configuration is environment-specific (not permissive in production)
- [ ] Static files are served from a CDN or reverse proxy, not directly from FastAPI
- [ ] Health check endpoint exists and is suitable for load balancer probes
- [ ] Application starts with `validate_on_startup()` to catch misconfigurations early

---

## Output Format

Save to `specs/security/audit-YYYY-MM-DD.md` with:

### 1. Executive Summary
- 3-sentence overall security posture assessment
- Count of findings by severity

### 2. Scorecard
| Category | Rating | Finding Summary |
|----------|--------|-----------------|
| Authentication | ✅/🟡/🟠/🔴 | one-line |
| Authorization & Tenant Isolation | ... | ... |
| ... | ... | ... |

### 3. Detailed Findings
For each finding:
- **Severity:** 🔴/🟠/🟡
- **File:** exact path and line number
- **Description:** what the vulnerability is
- **Impact:** what an attacker could do
- **Fix:** specific code change recommended

### 4. Passed Checks
List everything that passed — this is important for tracking coverage over time.

### 5. Recommendations
Prioritized list of fixes, grouped by severity.

## DO

- Read `specs/README.md` FIRST for the file index
- Check EVERY route handler for auth and tenant isolation
- Provide exact file paths and line numbers for findings
- Test IDOR by tracing the full code path from route → service → repository → query
- Check for secrets in the git history if possible: `git log --all -p -S "sk_live_" -- "*.py" "*.js"`

## DO NOT

- Skip any category — even if it looks fine, confirm it
- Assume Pydantic validation is sufficient without checking field constraints
- Modify any code — this is read-only
- Assume rate limiting is applied just because the limiter exists — verify each route
- Trust that security headers middleware is registered without checking main.py
```
