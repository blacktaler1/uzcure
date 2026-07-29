"""
AI Operation Restriction Check — CORE UTILITY
==============================================
DESIGN: Replaced patient-consent-gate (opt-in) with doctor-controlled
RESTRICT model (opt-out). Plan generation is NEVER blocked by missing
patient consent — only an explicit doctor/admin restriction flag stops it.

Consent records are preserved for GDPR/audit purposes but are NON-BLOCKING.
A missing or withdrawn consent is logged, not rejected.

M-06 NOTE: This is the LOW-LEVEL restriction check function.
Used by: app/api/deps/consent.py (FastAPI dependency wrapper)
Do NOT confuse with: app/api/endpoints/consent.py (HTTP endpoints)
"""

import logging
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.api.deps import get_current_user

logger = logging.getLogger(__name__)


async def check_ai_consent(consent_type: str, patient_id: int, db: Session) -> bool:
    """
    RESTRICT MODEL: Check if an AI operation is explicitly restricted.

    Previously: blocked if patient had NOT given consent (opt-in).
    Now:        blocks ONLY if patient record has is_withdrawn=True AND
                the relevant flag is explicitly False after withdrawal.
                In all other cases (no record, missing flag) → ALLOW.

    This ensures doctors can always generate plans. Consent records
    are informational / GDPR audit trail only.

    Args:
        consent_type: Type of AI operation (plan_generation, etc.)
        patient_id: Patient ID to check
        db: Database session

    Returns:
        True always (restriction is now opt-out, not opt-in)
    """
    from app.models.ai_consent import AIConsent

    try:
        consent = db.query(AIConsent).filter(AIConsent.patient_id == patient_id).first()

        if consent and consent.is_withdrawn:
            # Map consent_type → model column
            restriction_mapping = {
                "plan_generation": consent.ai_plan_generation,
                "exercise_recommendations": consent.ai_exercise_recommendations,
                "progress_analysis": consent.ai_progress_analysis,
                "feedback_processing": consent.ai_feedback_processing,
            }
            # Only block if explicitly restricted AFTER withdrawal
            # (flag was True then patient withdrew — doctor must re-confirm)
            # For plan_generation we NEVER block — doctor authority supersedes
            if consent_type != "plan_generation":
                restricted = restriction_mapping.get(consent_type)
                if restricted is False:
                    logger.warning(
                        "AI operation '%s' restricted for patient_id=%s "
                        "(consent withdrawn, flag=False). "
                        "Doctor override required.",
                        consent_type,
                        patient_id,
                    )
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "operation_restricted",
                            "message": (
                                f"Patient has restricted '{consent_type.replace('_', ' ')}'. "
                                "A doctor override or patient re-consent is required."
                            ),
                            "consent_type": consent_type,
                            "patient_id": patient_id,
                        },
                    )

        # Log consent status for audit — do NOT block
        if not consent:
            logger.info(
                "No consent record for patient_id=%s; operation '%s' proceeding "
                "(restrict model: missing record = allowed).",
                patient_id,
                consent_type,
            )
        else:
            flag_val = {
                "plan_generation": consent.ai_plan_generation,
                "exercise_recommendations": consent.ai_exercise_recommendations,
                "progress_analysis": consent.ai_progress_analysis,
                "feedback_processing": consent.ai_feedback_processing,
            }.get(consent_type, True)
            logger.debug(
                "Consent check patient_id=%s type='%s' flag=%s withdrawn=%s → ALLOWED",
                patient_id,
                consent_type,
                flag_val,
                consent.is_withdrawn,
            )

    except HTTPException:
        raise
    except Exception as exc:
        # Never let consent DB errors crash the pipeline
        logger.error(
            "Consent check DB error for patient_id=%s type='%s': %s — proceeding.",
            patient_id,
            consent_type,
            exc,
        )

    return True


def require_ai_consent(consent_type: str):
    """
    Dependency factory — restrict check (non-blocking for plan_generation).

    Usage:
        @router.post("/generate-plan")
        async def generate_plan(
            request: PlanRequest,
            _ok: bool = Depends(require_ai_consent("plan_generation")),
            db: Session = Depends(get_db)
        ):
            ...
    """

    async def consent_checker(
        patient_id: int = None,
        db: Session = Depends(get_db),
        current_user=Depends(get_current_user),
    ):
        if patient_id is None:
            if current_user.role == "patient":
                from app.models.patient import Patient

                patient = db.query(Patient).filter(Patient.user_id == current_user.id).first()
                if patient:
                    patient_id = patient.id

        if patient_id is None:
            # No patient context — skip restrict check, allow operation
            logger.info(
                "require_ai_consent('%s'): no patient_id resolved, skipping check.", consent_type
            )
            return True

        return await check_ai_consent(consent_type, patient_id, db)

    return consent_checker


class ConsentChecker:
    """
    Class-based restriction checker for complex scenarios.
    """

    def __init__(self, db: Session):
        self.db = db

    async def verify(self, patient_id: int, consent_type: str) -> bool:
        """Verify restriction — returns True or raises HTTPException."""
        return await check_ai_consent(consent_type, patient_id, self.db)

    def has_consent(self, patient_id: int, consent_type: str) -> bool:
        """
        Non-raising check. Returns True if NOT restricted, False if restricted.
        Under restrict model, default is True (allowed) unless explicitly restricted.
        """
        from app.models.ai_consent import AIConsent

        try:
            consent = self.db.query(AIConsent).filter(AIConsent.patient_id == patient_id).first()

            if not consent:
                return True  # No record → allowed

            if not consent.is_withdrawn:
                return True  # Active consent → allowed

            # Withdrawn: check specific flag
            mapping = {
                "plan_generation": consent.ai_plan_generation,
                "exercise_recommendations": consent.ai_exercise_recommendations,
                "progress_analysis": consent.ai_progress_analysis,
                "feedback_processing": consent.ai_feedback_processing,
            }
            flag = mapping.get(consent_type, True)
            # plan_generation always returns True (doctor authority)
            if consent_type == "plan_generation":
                return True
            return flag is not False  # None or True → allowed

        except Exception as exc:
            logger.error("has_consent DB error: %s — returning True.", exc)
            return True
