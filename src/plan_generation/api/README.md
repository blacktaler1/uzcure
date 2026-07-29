# Plan generation API

> **Note:** this is the original internal API document, reproduced unedited. It references the
> worker, repositories, and frontend dashboard, which live in the private repository and are not
> part of this public excerpt. See the root `README.md`.

Job-queue pattern. POST returns instantly with `job_id`; client polls until terminal state.

## Endpoints

### `POST /api/plans/jobs`

**Status:** 202 Accepted (always; non-202 = pre-validation error)

**Auth:** `Bearer <jwt>` + `X-Clinic-Slug` (or `X-Clinic-ID`) header. Doctor role + verified.

**Headers:**
- `Authorization: Bearer ...` — JWT, role=doctor, is_verified=true
- `X-Clinic-Slug: <slug>` (or `X-Clinic-ID: <id>`)
- `Idempotency-Key: <opaque>` (optional) — same key + same payload + same doctor → returns the existing job_id instead of creating a new one. Different content under same key → new job.

**Body** (`application/json`):
```json
{
  "patient_id": 42,
  "patient_gender": "female",
  "case_history_text": "72-year-old female, post-TKR rehab...",
  "document_id": null,
  "output_language": "uzbek",
  "clinician_override": false
}
```

`case_history_text` is required (20–50 000 chars, stripped). Other fields optional.

**Response:**
```json
{ "job_id": "1eff40b9...", "status": "queued" }
```

**Rate limit:** `AI_GENERATE_PLAN_LIMIT` (per user). Hits return 429 + Retry-After.

**Errors:**
| Code | When |
|---|---|
| 401 | Missing/invalid JWT |
| 403 | Doctor not verified |
| 422 | Body validation failed (extra field, length, enum) |
| 429 | Rate limit exceeded |

---

### `GET /api/plans/jobs/{job_id}`

**Status:** 200 / 404

**Auth:** same as POST. Job is tenant-scoped: a job created in clinic A cannot be read from clinic B (returns 404).

**Response:**
```json
{
  "job_id": "...",
  "status": "queued|running|succeeded|failed|cancelled",
  "progress_step": "guards|verdict|exercise_pool|drug_facts|llm_call|exercise_resolve|done|...",
  "result": { /* full GeneratedPlan when status=succeeded */ } | null,
  "error_code": "PATIENT_NOT_FOUND" | null,
  "error_message": "human-readable" | null,
  "created_at": "ISO 8601",
  "started_at": "ISO 8601" | null,
  "completed_at": "ISO 8601" | null
}
```

**Polling cadence:** every 3 seconds is the suggested default. Backoff to 5 s after 30 s of running. p50 latency 30–60 s; p95 ≤ 180 s.

---

### `DELETE /api/plans/jobs/{job_id}`

**Status:** 200 / 409

Cancels a `queued` or `running` job. Already-terminal jobs return 409.

**Response:**
```json
{ "job_id": "...", "status": "cancelled" }
```

---

### Approval lifecycle

- `POST /api/plans/jobs/{job_id}/approve` — doctor approves; backend persists the plan, returns `plan_id`.
- `POST /api/plans/jobs/{job_id}/reject` — doctor rejects with reason.
- `POST /api/plans/jobs/{job_id}/edit` — doctor saves edited result back onto the job.
- `POST /api/plans/{plan_id}/deliver` — releases approved plan to patient view.

---

## Status code matrix

| Path | Domain error | HTTP |
|---|---|---|
| (any) | `PatientNotFound` | 404 |
| (any) | `TenancyViolation` | 403 |
| (any) | `ConsentMissingError` | 403 |
| (any) | `PlanLimitExceeded` | 409 |
| (any) | `SafetyVerdictBlocked` | 422 |
| (worker) | `SchemaValidationError` | 502 (in job result) |
| (worker) | `LanguageComplianceError` | 502 (in job result) |
| (worker) | `LLMUpstreamError` | 502 (in job result) |
| (worker) | `GenerationCooldownActive` | 429 (in job result; route has cooldown gate too) |

The use case raises domain errors; the HTTP layer (`api/exception_handlers.py`) maps them. Worker-side errors land in `job.error_code` / `job.error_message` and the GET endpoint surfaces them.

## Observability

Every state transition logs JSON-structured events with `job_id`, `verdict`, `tokens_in`, `tokens_out`, `cost_usd`, `latency_ms`. PHI is forbidden in logs (see `_logging.py`).

## Frontend integration

- `services/doctor.service.ts`: `enqueueGeneratePlan`, `getPlanJobStatus`, `cancelPlanJob`
- Hook: `useEnqueuePlanJob` (mutation) + `usePlanJobStatus(jobId)` (query)
- See dashboard repo `src/features/doctor/rehab-plans/`.
