# UzCure — AI Rehabilitation Plan Generation

**Public code excerpt.** This repository contains a curated portion of the MyRehab production
codebase, published for the President Tech Award incubation program review. The full system is a
private repository.

> **Qisqacha (UZ):** MyRehab — reabilitatsiya klinikalari uchun platforma. Bu repoda uning eng
> murakkab qismi ochilgan: **sun'iy intellekt yordamida bemorga reabilitatsiya rejasini
> tayyorlash**. Asosiy tamoyil — AI **hech qachon bemorga to'g'ridan-to'g'ri chiqmaydi**. Model
> chiqargan har bir reja avval xavfsizlik tekshiruvidan o'tadi, keyin **shifokor tasdiqlaydi**.
> Shifokor tasdiqlamasa — bemor uni ko'rmaydi.

---

## The problem

An LLM can write a plausible-looking rehabilitation plan in seconds. It can also invent an exercise
that does not exist, prescribe a movement that is contraindicated for the patient's condition,
ignore a documented allergy, or silently produce the plan in the wrong language for a patient who
does not read it.

A rehabilitation platform therefore cannot simply "call the model and show the result."

## The approach

Plan generation is built as a **hexagonal (ports-and-adapters) module** with an explicitly enforced
safety pipeline. Three rules shape the design:

1. **The AI never reaches the patient directly.** Generation produces a *candidate*. A verified
   doctor must approve it. Only then is it delivered to the patient view. Critical safety findings
   require an explicit acknowledgement from the doctor before approval is even possible.

2. **The model may abstain.** The pipeline can return `ESCALATE`, `INSUFFICIENT`,
   `MONITORING_ONLY`, or `SPECIALIST_REQUIRED` instead of a plan. "I should not answer this" is a
   first-class outcome, not a failure.

3. **Every claim is checked against the patient record.** Exercises are grounded against a real
   clinical registry — an invented exercise is caught and repaired. Contraindications, documented
   allergies, and medication doses are reconciled against the patient's own data, not against the
   model's memory of it.

Output language is an enum, not a free-form string, and is verified after generation — a patient
who reads Uzbek never receives a Russian plan.

---

## What is in this repository

### `src/plan_generation/` — the hexagonal module

| Layer | Contents |
|---|---|
| `domain/` | Pure data and invariants, zero infrastructure imports. Pydantic v2 entities with `extra="forbid"`, frozen command object, domain error hierarchy, language detection. Layer purity is test-enforced in the private repo. |
| `application/` | `ports.py` — the interfaces the domain depends on. `safety_check.py` (38 KB) — the clinical safety pipeline. `patient_fidelity.py` — verification that the plan matches the actual patient record. `verdict_reconciler.py` — reconciles model verdict with deterministic checks. `approval_use_case.py` — the doctor approval lifecycle. |
| `infrastructure/` | `llm_factory.py` + `fallback_provider.py` — multi-provider routing with automatic failover. Anthropic and DeepSeek adapters. `cooldown_guard.py`, `language_check.py`, `ledger_writer.py`. |
| `api/` | FastAPI job-queue surface: `POST` returns a `job_id` immediately, client polls to a terminal state. Tenant-scoped, idempotency-key aware, rate-limited, with a documented domain-error → HTTP status matrix. |
| `_logging.py` | Structured JSON observability. **PHI is forbidden in logs** and this is enforced here. |

`api/README.md` and `domain/README.md` are the original internal engineering documents, included
unedited — they show how the module is actually specified and maintained.

### `src/security/` — clinical data protection

| File | What it does |
|---|---|
| `malware_scan.py` | Magic-byte signature validation + ClamAV scan for patient document uploads. Degrades to a hash blacklist when ClamAV is unreachable; an unverifiable file is quarantined, not accepted. |
| `phi_deid.py` | Protected health information de-identification. |
| `upload_safety.py` | Upload validation gate. |
| `secure_logging.py` | Log redaction. |
| `consent_check.py` | Patient consent enforcement. |

### `tests/` — 176 test functions

Including property-based safety-invariant tests, grounding-repair tests (invented exercise
detection), documented-allergy handling, medication dose reconciliation, escalation/abstention
behaviour, tenant isolation, idempotency, and malware/PHI tests.

**~6,200 lines of source and ~3,200 lines of tests.**

## What is deliberately not in this repository

The full private repository additionally contains: the main orchestration use case
(`application/use_case.py`), the prompt builder and LLM output schema, the plan writer, the clinical
exercise registries (MSK, neuro, cardiac, pulmonary, geriatric, pediatric, rheumatology, chronic
pain, women's health — several megabytes of curated clinical content), the evidence retriever, the
notification engine, the MDT case-conference module, the doctor reward ledger, 226 Alembic
migrations, and the Vite frontend.

The registries and prompt engineering are the product's core intellectual property. This excerpt was
chosen to show architecture and safety engineering instead.

---

## Architecture

```
  Doctor submits case history
            │
            ▼
   ┌────────────────┐
   │  API (FastAPI) │  auth · tenant scope · consent · rate limit · idempotency
   └───────┬────────┘
           │  202 Accepted + job_id        ┌──────────────────────┐
           ▼                               │  client polls status │
   ┌────────────────┐                      └──────────────────────┘
   │  Job queue     │
   └───────┬────────┘
           ▼
   ┌───────────────────────────────────────────────────┐
   │  APPLICATION LAYER                                │
   │                                                   │
   │   guards ──▶ verdict ──▶ exercise pool ──▶        │
   │   drug facts ──▶ LLM call ──▶ exercise resolve    │
   │                                                   │
   │   ┌─────────────────────────────────────────┐     │
   │   │  safety_check      contraindications,   │     │
   │   │                    allergies, doses      │     │
   │   │  patient_fidelity  plan vs real record  │     │
   │   │  verdict_reconcile model vs determinist.│     │
   │   │  language_check    output language      │     │
   │   └─────────────────────────────────────────┘     │
   └───────────────────────┬───────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      abstain / escalate         candidate plan
      (no plan produced)                │
                                        ▼
                            ┌───────────────────────┐
                            │  DOCTOR APPROVAL      │  critical findings
                            │  (verified doctor)    │  require explicit ack
                            └───────────┬───────────┘
                                        │ approved
                                        ▼
                                  Patient view
```

The domain layer imports nothing from infrastructure. Providers (Anthropic, OpenAI, Gemini,
DeepSeek) are adapters behind a port, selected by `llm_factory.py` with automatic fallback — the
business rules do not know which model answered.

## Stack

Python 3.12 · FastAPI · PostgreSQL · SQLAlchemy · Alembic · Pydantic v2 · pytest (incl.
property-based tests) · ruff · mypy

Frontend: Vite + JS. Deployment: Azure App Service (container) + Azure Database for PostgreSQL
Flexible Server + Azure Static Web Apps. CI: migration-guard, ruff, mypy, pytest.

## Running this excerpt

These files are lifted unmodified from the production tree, so their imports still reference the
full `app.features.plan_generation.*` package. **This excerpt is meant to be read, not executed** —
the modules it depends on live in the private repository, where the suite runs in CI under
migration-guard, `ruff`, `mypy`, and `pytest`.

Reviewers who want to run the real system against the full test suite can request access to the
private repository.

## Where to start reading

1. `src/plan_generation/domain/README.md` — the layer contract, written before the code.
2. `src/plan_generation/application/ports.py` — the seam between business rules and the outside world.
3. `src/plan_generation/application/safety_check.py` — the clinical safety pipeline.
4. `tests/application/test_safety_invariants_property.py` — property-based proof of the safety invariants.
5. `tests/application/test_invented_exercise_grounding.py` — what happens when the model makes something up.
