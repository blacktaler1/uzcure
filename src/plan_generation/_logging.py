"""Structured logging helpers for plan generation v2.

Strict contract:
  - Every plan_v2 log line is JSON-shaped via the project logger.
  - PHI is NEVER allowed in log payloads:
      * patient name (decrypted)
      * patient phone
      * raw case_history_text
      * patient address / DOB beyond age_years
  - Allowed identifiers:
      * job_id (UUID, server-generated)
      * doctor_id (clinician)
      * clinic_id (tenancy)
      * patient_id (numeric, opaque)
      * stage / step name
      * model id, token counts, cost, latency

This module exposes a thin wrapper so log statements remain consistent and
auditable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("plan_v2")


# Field names that, if present in a log payload, should be dropped.
# These are PHI-bearing or otherwise too revealing for routine logs.
# Categories below come from the PHI section of docs/db_schema_audit.md.
_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        # Patient identity
        "patient_name",
        "display_name",
        "name",
        "phone",
        "phone_number",
        "address",
        "dob",
        "date_of_birth",
        # Free-text clinical narrative
        "case_history_text",
        "case_text",
        "raw_text",
        "diagnosis_text",
        "diagnosis",
        "notes",
        # Patient medication PHI (patient_medications table)
        "medication_name",
        "medication",
        "dose",
        "dosage",
        "drug_generic_name",
        "drug_name",  # patient-bound; oracle's drug_name in admin context can be relogged with explicit consent
        "rehab_precautions",
        "patient_explanation",
        # Allergies (patient_allergies)
        "allergen",
        # Lab results (patient_lab_results)
        "lab_value",
        "test_name",
        # Vital signs (patient_vital_signs)
        "systolic_bp",
        "diastolic_bp",
        "bp_systolic",
        "bp_diastolic",
        "heart_rate",
        "respiratory_rate",
        "temperature",
        "spo2",
        "weight_kg",
        "height_cm",
        "bmi",
    }
)


def _strip_phi(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in _FORBIDDEN_KEYS}


def event(event_name: str, **fields: Any) -> None:
    """Emit a structured plan_v2 log event.

    Example:
        event("job_started", job_id=jid, doctor_id=did, clinic_id=cid)

    PHI keys are stripped automatically. Use `event_name` as a stable token
    that grafana / log search can pivot on.
    """
    payload = _strip_phi(fields)
    payload["event"] = f"plan_v2.{event_name}"
    logger.info("plan_v2.%s", event_name, extra={"plan_v2": payload})


def warn(event_name: str, **fields: Any) -> None:
    payload = _strip_phi(fields)
    payload["event"] = f"plan_v2.{event_name}"
    logger.warning("plan_v2.%s", event_name, extra={"plan_v2": payload})


def error(event_name: str, **fields: Any) -> None:
    payload = _strip_phi(fields)
    payload["event"] = f"plan_v2.{event_name}"
    logger.error("plan_v2.%s", event_name, extra={"plan_v2": payload})


__all__ = ["event", "warn", "error", "_FORBIDDEN_KEYS"]
