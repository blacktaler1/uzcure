"""Documented-allergy surfacing + cross-check — deterministic, universal.

A patient's documented allergy must be (a) SURFACED in the plan the doctor/patient
see and (b) CROSS-CHECKED against what the plan prescribes — honoured whether it
was recorded in the structured allergy field OR stated in the free-text case. The
free-text extractor is universal (marker-based, ANY substance — never a curated
drug list). Injection is deterministic (never the LLM, which could drop it) as a
localized ``SafetyAlert`` that both the composer review and the patient plan view
already render.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.features.plan_generation.application.ports import PatientSnapshot, RegistryExercise
from app.features.plan_generation.application.safety_check import (
    extract_case_allergens,
    merge_allergens,
    surface_documented_allergies,
)
from app.features.plan_generation.application.use_case import GeneratePlanUseCase
from app.features.plan_generation.domain import PlanLanguage, SafetyAlert
from app.features.plan_generation.infrastructure.prompt_builder import build_prompt


def _plan(lang: str, alerts=None):
    return SimpleNamespace(output_language=lang, safety_alerts=list(alerts or []))


def _patient(allergies):
    return PatientSnapshot(
        patient_id=1,
        display_name="x",
        primary_language=PlanLanguage.RU,
        gender="female",
        age_years=69,
        diagnosis="Status post left THA",
        allergies=allergies,
        comorbidities=[],
        known_medications=[],
    )


# ── surfacing (unified allergen list) ───────────────────────────────────────


@pytest.mark.parametrize(
    "lang,expected_label",
    [
        ("en", "Documented allergy"),
        ("ru", "Задокументированная аллергия"),
        ("uz", "Hujjatlashtirilgan allergiya"),
        ("uz-cyrl", "Ҳужжатлаштирилган аллергия"),
    ],
)
def test_allergy_surfaced_localized(lang, expected_label):
    p = _plan(lang)
    added = surface_documented_allergies(p, ["diclofenac"])
    assert added == 1
    assert p.safety_alerts[0].alert == expected_label
    assert "diclofenac" in p.safety_alerts[0].detail


def test_unknown_language_falls_back_to_english_but_still_surfaces():
    p = _plan("fr")
    assert surface_documented_allergies(p, ["diclofenac"]) == 1
    assert p.safety_alerts[0].alert == "Documented allergy"


def test_multiple_allergens_each_surfaced():
    p = _plan("en")
    added = surface_documented_allergies(p, ["diclofenac", "penicillin"])
    assert added == 2
    joined = " ".join(sa.detail for sa in p.safety_alerts).lower()
    assert "diclofenac" in joined and "penicillin" in joined


def test_no_duplicate_when_allergen_already_named():
    existing = SafetyAlert(alert="NSAID caution", detail="Avoid diclofenac — allergy.")
    p = _plan("en", [existing])
    assert surface_documented_allergies(p, ["diclofenac"]) == 0
    assert len(p.safety_alerts) == 1


def test_noop_when_no_allergen():
    assert surface_documented_allergies(_plan("en"), []) == 0
    assert surface_documented_allergies(_plan("en"), None) == 0


# ── free-text extractor (universal / substance-agnostic) ────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Аллергия на диклофенак — генерализованная крапивница и отёк губ.", ["диклофенак"]),
        ("Allergy to penicillin (rash).", ["penicillin"]),
        ("Allergic to aspirin and sulfa.", ["aspirin", "sulfa"]),  # multi-allergen, both kept
        ("аллергию на пенициллин и морепродукты", ["пенициллин", "морепродукты"]),
        ("Аллергия: латекс", ["латекс"]),
        ("Аллергия на аспирин, сульфаниламиды; отёк.", ["аспирин", "сульфаниламиды"]),
        ("История сахарного диабета, без аллергий.", []),  # negated
        ("No known drug allergies.", []),
        ("No allergy to penicillin.", []),  # negated
        ("Аллергий нет.", []),
        ("", []),
        (None, []),
    ],
)
def test_extract_case_allergens_universal(text, expected):
    assert extract_case_allergens(text) == expected


def test_merge_allergens_unions_record_and_case_deduped():
    # Structured record has one allergen; the case narrative states another + a
    # duplicate of the first (different case) — merged, de-duplicated.
    patient = _patient(["Diclofenac"])
    merged = merge_allergens(patient, "Also аллергия на пенициллин and allergy to diclofenac.")
    low = [m.lower() for m in merged]
    assert "diclofenac" in low and "пенициллин" in low
    assert len(low) == len(set(low)), "merge must de-duplicate"


def test_merge_allergens_case_only_when_no_structured_record():
    # Case-history-only run: no structured patient, allergy only in free text.
    assert merge_allergens(None, "Аллергия на диклофенак — крапивница.") == ["диклофенак"]


# ── end-to-end through execute() ────────────────────────────────────────────


def _registry_ex(eid: str) -> RegistryExercise:
    return RegistryExercise(
        exercise_id=eid,
        canonical_name=eid.replace("_", " ").title(),
        display_name_uz=None,
        display_name_ru=None,
        display_name_en=None,
        specialty="msk",
        body_region="LL",
        category="strengthening",
        default_sets="3",
        default_reps="10",
        contraindications=[],
        has_verified_video=True,
        video_url="https://x",
    )


@pytest.mark.asyncio
async def test_structured_allergy_surfaced_end_to_end(
    sample_command,
    sample_plan_payload,
    fake_exercise_repo,
    fake_cooldown,
):
    """A patient with a STRUCTURED allergy → surfaced in the generated plan."""
    from tests.features.plan_generation.conftest import FakeLLMProvider, FakePatientRepo

    snapshot = _patient(["diclofenac"])
    sample_plan_payload["phases"][0]["exercises"][0]["exercise_id"] = "MSK_KNEE_FLEX_001"
    fake_exercise_repo.pool = [_registry_ex("MSK_KNEE_FLEX_001")]

    cmd = sample_command.model_copy(update={"patient_id": 1})
    uc = GeneratePlanUseCase(
        patient_repo=FakePatientRepo(snapshot=snapshot),
        exercise_repo=fake_exercise_repo,
        llm=FakeLLMProvider(canned_response=sample_plan_payload),
        cooldown=fake_cooldown,
        prompt_builder=build_prompt,
        language_checker=lambda text, lang: True,
    )
    outcome = await uc.execute(cmd, job_id="allergy-structured")
    surfaced = [
        sa for sa in outcome.plan.safety_alerts if "diclofenac" in (sa.detail or "").lower()
    ]
    assert surfaced, "structured allergy not surfaced"


@pytest.mark.asyncio
async def test_free_text_allergy_surfaced_end_to_end_case_only(
    sample_command,
    sample_plan_payload,
    fake_patient_repo,
    fake_exercise_repo,
    fake_cooldown,
):
    """Case-history-only run whose FREE TEXT states an allergy → still surfaced.

    Mirrors the audited case: the diclofenac allergy lives in the narrative, not a
    structured field. It must still appear in the plan.
    """
    from tests.features.plan_generation.conftest import FakeLLMProvider

    cmd = sample_command.model_copy(
        update={
            "patient_id": None,
            "case_history_text": (
                "69-year-old, post left THA day 6. Аллергия на диклофенак — крапивница."
            ),
        }
    )
    sample_plan_payload["phases"][0]["exercises"][0]["exercise_id"] = "MSK_KNEE_FLEX_001"
    fake_exercise_repo.pool = [_registry_ex("MSK_KNEE_FLEX_001")]

    uc = GeneratePlanUseCase(
        patient_repo=fake_patient_repo,
        exercise_repo=fake_exercise_repo,
        llm=FakeLLMProvider(canned_response=sample_plan_payload),
        cooldown=fake_cooldown,
        prompt_builder=build_prompt,
        language_checker=lambda text, lang: True,
    )
    outcome = await uc.execute(cmd, job_id="allergy-freetext")
    surfaced = [
        sa for sa in outcome.plan.safety_alerts if "диклофенак" in (sa.detail or "").lower()
    ]
    assert surfaced, "free-text allergy 'диклофенак' was not surfaced"


@pytest.mark.asyncio
async def test_free_text_allergen_cross_checked_against_prescription(
    sample_command,
    sample_plan_payload,
    fake_patient_repo,
    fake_exercise_repo,
    fake_cooldown,
):
    """A free-text allergen the plan PRESCRIBES → recorded as an allergy hit.

    Proves the free-text allergen reaches the cross-check (not just surfacing),
    end-to-end and same-script (English case + English med name).
    """
    from tests.features.plan_generation.conftest import FakeLLMProvider

    cmd = sample_command.model_copy(
        update={
            "patient_id": None,
            "case_history_text": "Post left THA. Allergic to diclofenac (urticaria).",
        }
    )
    sample_plan_payload["phases"][0]["exercises"][0]["exercise_id"] = "MSK_KNEE_FLEX_001"
    sample_plan_payload["medication_schedule"] = [
        {
            "time": "morning",
            "medication": "Diclofenac 50 mg",
            "dose": "50 mg",
            "instructions": "with food",
            "duration": "prn",
        }
    ]
    fake_exercise_repo.pool = [_registry_ex("MSK_KNEE_FLEX_001")]

    uc = GeneratePlanUseCase(
        patient_repo=fake_patient_repo,
        exercise_repo=fake_exercise_repo,
        llm=FakeLLMProvider(canned_response=sample_plan_payload),
        cooldown=fake_cooldown,
        prompt_builder=build_prompt,
        language_checker=lambda text, lang: True,
    )
    outcome = await uc.execute(cmd, job_id="allergy-crosscheck")
    hits = (outcome.safety_meta or {}).get("allergy_hits") or []
    assert any("diclofenac" in (h.get("allergen") or "").lower() for h in hits), (
        f"free-text allergen not cross-checked against the prescribed med; hits={hits}"
    )
