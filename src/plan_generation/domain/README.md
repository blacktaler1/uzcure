# Domain Layer — Plan Generation

> **Note:** this is the original internal document, reproduced unedited. It references files that
> live in the private repository and are not part of this public excerpt (`verdict.py`,
> `application/use_case.py`, `infrastructure/output_schema.py`, and the domain test suite).
> See the root `README.md` for what this excerpt does and does not include.

Pure data + invariants. Zero infrastructure dependencies.

## What lives here

| File | Owns |
|---|---|
| `entities.py` | Pydantic v2 models for the input command and the full output plan + every nested entity |
| `verdict.py` | `Verdict` enum (LLM_FULL, ESCALATE, INSUFFICIENT, MONITORING_ONLY, SPECIALIST_REQUIRED) |
| `errors.py` | `DomainError` hierarchy. Use case raises these; presentation layer maps them to HTTP. |
| `__init__.py` | Public re-exports |

## Hard rules

1. **No SQLAlchemy, FastAPI, Anthropic SDK, redis, or `app.db` / `app.models` / `app.api` imports.** Enforced by `tests/features/plan_generation/test_layer_purity.py`.
2. **All Pydantic models use `extra="forbid"`.** Adding an unknown field is a contract change; reject at validation.
3. **`GeneratePlanCommand` is `frozen=True`.** Once built, it cannot mutate. Use case treats it as a value object.
4. **Output language is enum, not free-form string.** Every code path that reads `output_language` works on `PlanLanguage` instances.
5. **Internal-only fields are documented.** `Exercise.exercise_id`, `Exercise.ai_suggested`, `DrugInteraction.source` exist on the entity but are stripped from the public API response by `GeneratedPlan.to_api_dict()`.

## Mutability policy (deviation from plan Step 34)

The plan-spec target was "all domain entities frozen, use `model_copy()` for any mutation." We deliberately keep `Exercise` (and the plan tree it lives in) mutable for the post-LLM resolve loop in `application/use_case.py:_resolve_exercises`. Reason:

- The resolver attaches `has_video`, `ai_suggested`, and clears `exercise_id` per-exercise after a registry lookup.
- Frozen rebuild would require reconstructing every Phase + GeneratedPlan from the leaves up — verbose, no clinical benefit.
- The plan never escapes the use case mutable; every external boundary (API response, job result) calls `to_api_dict()` which deep-strips internal fields.

Frozen: `GeneratePlanCommand` only.
Forbid-extras: every entity.
This is a documented exception. Tests confirm the public surface is stable.

## Public API contract

`GeneratedPlan.to_api_dict()` returns the exact public shape the dashboard renders. Any public-facing change must:
1. Update the entity field.
2. Update `infrastructure/output_schema.py` (Anthropic tool input schema must agree).
3. Update API DTO if exposed.
4. Update one `tests/features/plan_generation/domain/test_plan.py` assertion.

## How to extend

Add a new entity:
1. Define the Pydantic class in `entities.py`. Always `extra="forbid"`.
2. If the class is reachable through `GeneratedPlan`, also add it to `output_schema.py` (Anthropic tool schema mirror).
3. Add a unit test in `tests/features/plan_generation/domain/test_plan.py`.
4. Re-export from `domain/__init__.py`.

Add a new domain exception:
1. Subclass `DomainError` in `errors.py`. Set `code: str` class attribute.
2. Re-export from `domain/__init__.py`.
3. Map to HTTP in `api/exception_handlers.py` `_STATUS_FOR_CODE`.

## Internal-vs-public field map

| Entity | Public | Internal (stripped) |
|---|---|---|
| `Exercise` | name, sets, reps, duration_minutes, frequency, notes, has_video | exercise_id, ai_suggested |
| `DrugInteraction` | title, description, severity | source ("deterministic" \| "ai_extension") |
| `GeneratedPlan` | all 10 top-level keys per API contract | (none) |

## Why this layer exists

A clean domain layer:
- Lets us swap LLM providers, DBs, message brokers without touching business rules.
- Lets us unit-test plan-shape invariants without spinning up Postgres or hitting Anthropic.
- Documents the medical safety contract (frozen command, forbidden extras, language guarantee) in pure Python types.

## Testing

```bash
pytest tests/features/plan_generation/domain/ -q
pytest tests/features/plan_generation/test_layer_purity.py -q
```

Coverage target ≥ 95%. Domain layer rarely changes — every change deserves a test.
