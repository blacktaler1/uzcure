"""DI providers for plan_v2 API endpoints.

Centralizes construction of repos + use case so routes stay thin and
test overrides land at a single point. FastAPI `Depends(...)` plumbs
each provider into the endpoint signature; tests substitute via
`app.dependency_overrides`.
"""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.features.plan_generation.application.ports import (
    CooldownGuard,
    ExerciseRepo,
    JobRepo,
    LLMProvider,
    PatientRepo,
)
from app.features.plan_generation.application.use_case import GeneratePlanUseCase
from app.features.plan_generation.infrastructure.cooldown_guard import (
    FailureCooldownGuard,
)
from app.features.plan_generation.infrastructure.llm_factory import build_llm_provider
from app.features.plan_generation.infrastructure.exercise_repo import SqlExerciseRepo
from app.features.plan_generation.infrastructure.evidence_retriever import (
    build_evidence_retriever,
    build_pmid_verifier,
)
from app.features.plan_generation.infrastructure.job_repo import SqlJobRepo
from app.features.plan_generation.infrastructure.language_check import looks_compliant
from app.features.plan_generation.infrastructure.patient_repo import SqlPatientRepo
from app.features.plan_generation.infrastructure.prompt_builder import build_prompt


def get_job_repo(db: Session = Depends(get_db)) -> JobRepo:
    return SqlJobRepo(db)


def get_patient_repo(db: Session = Depends(get_db)) -> PatientRepo:
    return SqlPatientRepo(db)


def get_exercise_repo(db: Session = Depends(get_db)) -> ExerciseRepo:
    return SqlExerciseRepo(db)


def get_llm_provider() -> LLMProvider:
    """Build configured provider stack: Anthropic primary, OpenAI fallback
    when enabled and keyed. Per-request construction is cheap; both SDKs
    keep their HTTP pools internally."""
    return build_llm_provider()


def get_cooldown_guard() -> CooldownGuard:
    return FailureCooldownGuard()


def get_use_case(
    patient_repo: PatientRepo = Depends(get_patient_repo),
    exercise_repo: ExerciseRepo = Depends(get_exercise_repo),
    llm: LLMProvider = Depends(get_llm_provider),
    cooldown: CooldownGuard = Depends(get_cooldown_guard),
    db: Session = Depends(get_db),
) -> GeneratePlanUseCase:
    return GeneratePlanUseCase(
        patient_repo=patient_repo,
        exercise_repo=exercise_repo,
        llm=llm,
        cooldown=cooldown,
        prompt_builder=build_prompt,
        language_checker=looks_compliant,
        evidence_retriever=build_evidence_retriever(db),
        pmid_verifier=build_pmid_verifier(),
    )
