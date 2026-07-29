"""Universal operated/affected-side fidelity gate.

Deterministic, model-independent, condition-agnostic. Catches the external-audit
failure: an LLM that prescribes contralateral exercises (case = left hip, plan =
right-side exercises). Multilingual (en/ru/uz); flags ONLY an unambiguous
contradiction; never trips on bilateral / unspecified / both-sides / no-side.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.features.plan_generation.application.patient_fidelity import (
    check_fabricated_labs,
    check_monitoring_bounds,
    check_operated_side_fidelity,
    check_stopped_medication_scheduled,
    check_weight_bearing_fidelity,
)


class _PlanJson:
    """Minimal plan stub exposing model_dump_json (what the gate scans)."""

    def __init__(self, text: str) -> None:
        self._text = text

    def model_dump_json(self) -> str:
        return self._text


def _plan(*lateralities: str):
    """A minimal plan whose single phase carries one exercise per laterality."""
    exercises = [SimpleNamespace(laterality=lat) for lat in lateralities]
    return SimpleNamespace(phases=[SimpleNamespace(exercises=exercises)])


# ── the audited failure, both directions, all three languages ───────────────


@pytest.mark.parametrize(
    "case_text",
    [
        "Status post left total hip arthroplasty.",  # en
        "Состояние после тотального эндопротезирования левого ТБС.",  # ru
        "Chap son-bo'g'im protezidan keyingi holat.",  # uz
    ],
)
def test_case_left_plan_right_is_flagged(case_text):
    v = check_operated_side_fidelity(_plan("right", "bilateral"), case_text)
    assert v is not None
    assert v["case_side"] == "left"
    assert v["conflicting"] == ["right"]


@pytest.mark.parametrize(
    "case_text",
    [
        "Right total knee arthroplasty, POD 3.",  # en
        "Артропластика правого коленного сустава.",  # ru
        "O'ng tizza bo'g'imi artroplastikasi.",  # uz
    ],
)
def test_case_right_plan_left_is_flagged(case_text):
    v = check_operated_side_fidelity(_plan("left"), case_text)
    assert v is not None
    assert v["case_side"] == "right"
    assert v["conflicting"] == ["left"]


# ── must NOT trip (no false positives) ──────────────────────────────────────


def test_matching_side_passes():
    assert check_operated_side_fidelity(_plan("left"), "Left hip replacement.") is None


def test_bilateral_and_unspecified_never_trip():
    assert check_operated_side_fidelity(_plan("bilateral", ""), "Left hip replacement.") is None


def test_no_side_in_case_never_trips():
    # A generic case with no documented side must not flag any plan laterality.
    assert check_operated_side_fidelity(_plan("left", "right"), "Chronic low back pain.") is None


def test_both_sides_in_case_is_ambiguous_and_not_flagged():
    v = check_operated_side_fidelity(_plan("right"), "Bilateral knee OA, left worse than right.")
    assert v is None


def test_russian_non_anatomical_pravo_is_not_a_side():
    # "право" (a legal right) / "правило" (a rule) must NOT read as the RIGHT side.
    v = check_operated_side_fidelity(
        _plan("right"), "Пациент имеет право отказаться; правило клиники."
    )
    assert v is None


def test_empty_inputs_are_safe():
    assert check_operated_side_fidelity(_plan(), "") is None
    assert check_operated_side_fidelity(_plan("left"), "") is None


# ── the escalation wiring emits the PHI-free reason code ─────────────────────


def test_build_escalation_reasons_emits_side_mismatch_code():
    from app.features.plan_generation.application.use_case import _build_escalation_reasons
    from app.features.plan_generation.application.safety_check import SafetyCheckResult

    plan = SimpleNamespace(verdict=None, drug_interactions=[])
    safety = SafetyCheckResult()
    reasons = _build_escalation_reasons(
        plan,
        safety,
        operated_side_mismatch={
            "case_side": "left",
            "plan_sides": ["right"],
            "conflicting": ["right"],
        },
    )
    assert {"code": "operated_side_mismatch"} in reasons
    # And absent when there is no mismatch.
    assert not any(
        r.get("code") == "operated_side_mismatch" for r in _build_escalation_reasons(plan, safety)
    )


# ── fabricated-lab gate ─────────────────────────────────────────────────────


def test_invented_lab_value_not_in_case_is_flagged():
    plan = _PlanJson('{"summary": "Hb 104 g/L, CRP 18 mg/L; HbA1c 8.1%"}')
    invented = check_fabricated_labs(plan, "Diabetes, hypertension. No labs provided.")
    assert set(invented) == {"hemoglobin", "crp", "hba1c"}


def test_lab_present_in_case_is_trusted():
    # The case gives hemoglobin → the plan echoing it is NOT a fabrication.
    plan = _PlanJson('{"summary": "Hb 104, D-dimer 980"}')
    invented = check_fabricated_labs(plan, "Гемоглобин 105 г/л. CBC done.")
    assert "hemoglobin" not in invented
    assert "d_dimer" in invented  # d-dimer still invented (case never gave it)


def test_generic_lab_mention_without_a_number_is_not_a_fabrication():
    plan = _PlanJson('{"note": "Monitor renal function and platelets as clinically indicated."}')
    assert check_fabricated_labs(plan, "Post-op knee.") == []


def test_russian_invented_lab_is_flagged():
    plan = _PlanJson('{"summary": "СРБ 18 мг/л, СОЭ 32"}')
    invented = check_fabricated_labs(
        plan, "Состояние после эндопротезирования. Анализы не приложены."
    )
    assert set(invented) == {"crp", "esr"}


def test_uzbek_invented_lab_is_flagged():
    """#7 (universality): the lab lexicon carried EN + RU forms but not the
    Uzbek-Latin transliterations of the SAME terms already in it (gemoglobin /
    kreatinin / kaliy / natriy / leykotsit / trombotsit). An Uzbek plan that
    invents a lab value using Uzbek wording therefore slipped the gate. Uzbek is
    a first-class output language, so these must be covered."""
    plan = _PlanJson('{"summary": "kreatinin 250 mkmol/l, kaliy 6.5 mmol/l, gemoglobin 85 g/l"}')
    invented = check_fabricated_labs(
        plan, "Endoprotezlashdan keyingi holat. Tahlillar taqdim etilmagan."
    )
    assert {"creatinine", "potassium", "hemoglobin"}.issubset(set(invented)), invented


def test_uzbek_lab_present_in_case_is_trusted():
    """Symmetry: a lab the Uzbek case names (any value) is trusted, not flagged."""
    plan = _PlanJson('{"summary": "kaliy 6.5 mmol/l"}')
    invented = check_fabricated_labs(plan, "Kaliy 5.9 mmol/l topshirilgan.")
    assert "potassium" not in invented


def test_no_labs_in_plan_is_empty():
    plan = _PlanJson('{"phases": [{"exercises": [{"name": "Ankle pumps", "dosage": "3x15"}]}]}')
    assert check_fabricated_labs(plan, "Left hip arthroplasty.") == []


def test_build_escalation_reasons_emits_fabricated_data_code():
    from app.features.plan_generation.application.safety_check import SafetyCheckResult
    from app.features.plan_generation.application.use_case import _build_escalation_reasons

    plan = SimpleNamespace(verdict=None, drug_interactions=[])
    reasons = _build_escalation_reasons(
        plan, SafetyCheckResult(), fabricated_labs=["hemoglobin", "crp"]
    )
    assert {"code": "fabricated_patient_data", "count": 2} in reasons


# ── multilingual PLAN laterality (audit #3): the plan side is localised ──────
# Under the hard output-language contract the plan's `laterality` arrives in the
# plan language ("левая", "chap", "o'ng"), not the English literals. The gate
# must normalise it through the same lexicon or it silently never flags.


@pytest.mark.parametrize(
    "plan_lat",
    ["правая", "правый", "o'ng", "oʻng"],  # ru/uz "right"
)
def test_case_left_plan_localised_right_is_flagged(plan_lat):
    v = check_operated_side_fidelity(_plan(plan_lat), "Status post left total hip arthroplasty.")
    assert v is not None
    assert v["case_side"] == "left"
    assert v["conflicting"] == ["right"]


@pytest.mark.parametrize(
    "plan_lat",
    ["левая", "левый", "chap"],  # ru/uz "left"
)
def test_case_right_plan_localised_left_is_flagged(plan_lat):
    v = check_operated_side_fidelity(_plan(plan_lat), "Right total knee arthroplasty.")
    assert v is not None
    assert v["case_side"] == "right"
    assert v["conflicting"] == ["left"]


def test_localised_bilateral_plan_side_does_not_trip():
    # A laterality naming both sides (or neither) is never a mismatch.
    assert check_operated_side_fidelity(_plan("bilateral"), "Left hip arthroplasty.") is None
    assert check_operated_side_fidelity(_plan(""), "Left hip arthroplasty.") is None


# ── fidelity corpus (audit #1): the side lives in the STRUCTURED diagnosis ───
# The free-text case may omit the side while the clinician-entered diagnosis
# names it. build_fidelity_case_text folds diagnosis + comorbidities into the
# corpus so the gate still fires — closing the reopened-bug path.


def test_side_only_in_diagnosis_is_still_checked():
    from app.features.plan_generation.application.patient_fidelity import build_fidelity_case_text

    # Free text has NO side; the structured diagnosis does.
    corpus = build_fidelity_case_text(
        "Post-operative rehabilitation, POD 5. Labs not attached.",
        diagnosis="Состояние после тотального эндопротезирования левого ТБС",
        comorbidities=["гипертония"],
    )
    v = check_operated_side_fidelity(_plan("right"), corpus)
    assert v is not None
    assert v["case_side"] == "left"
    assert v["conflicting"] == ["right"]


def test_corpus_builder_handles_missing_structured_fields():
    from app.features.plan_generation.application.patient_fidelity import build_fidelity_case_text

    assert build_fidelity_case_text("case only", diagnosis=None, comorbidities=None) == "case only"
    assert build_fidelity_case_text(None, diagnosis="Left TKA") == "Left TKA"
    assert build_fidelity_case_text("", diagnosis="", comorbidities=[]) == ""


def test_lab_documented_in_comorbidities_is_not_fabricated():
    # A lab the record documents (in comorbidities) must NOT be flagged invented.
    from app.features.plan_generation.application.patient_fidelity import build_fidelity_case_text

    corpus = build_fidelity_case_text(
        "Post-op knee rehab.",
        diagnosis="Osteoarthritis",
        comorbidities=["Anemia, hemoglobin 104 g/L documented on admission"],
    )
    plan = _PlanJson('{"summary": "Given hemoglobin 104, monitor for fatigue."}')
    assert check_fabricated_labs(plan, corpus) == []


# ── omitted-medication gate (audit #4): a current drug dropped from the plan ─


def _plan_with_meds(*med_names: str):
    meds = [SimpleNamespace(medication=n) for n in med_names]
    return SimpleNamespace(medication_schedule=list(meds))


def test_current_medication_missing_from_schedule_is_flagged():
    from app.features.plan_generation.application.patient_fidelity import check_medication_omissions

    plan = _plan_with_meds("Rivaroxaban 10 mg", "Amlodipine 5 mg")
    # Metformin was a known med but never scheduled → omitted.
    omitted = check_medication_omissions(
        plan, ["Rivaroxaban 10mg", "Metformin 500mg", "Amlodipine"]
    )
    assert omitted == ["metformin"]


def test_all_medications_present_is_not_flagged():
    from app.features.plan_generation.application.patient_fidelity import check_medication_omissions

    plan = _plan_with_meds("Rivaroxaban 10 mg", "Metformin 500 mg")
    assert check_medication_omissions(plan, ["Rivaroxaban", "Metformin 500mg"]) == []


def test_medication_omissions_bidirectional_and_dose_insensitive():
    from app.features.plan_generation.application.patient_fidelity import check_medication_omissions

    # Plan lists the drug with a brand-ish suffix; known list is the plain name.
    plan = _plan_with_meds("insulin glargine 20 units")
    assert check_medication_omissions(plan, ["insulin glargine"]) == []


def test_medication_omissions_empty_inputs():
    from app.features.plan_generation.application.patient_fidelity import check_medication_omissions

    assert check_medication_omissions(_plan_with_meds(), []) == []
    assert check_medication_omissions(_plan_with_meds("Aspirin"), None) == []


def test_build_escalation_reasons_emits_medication_omitted_code():
    from app.features.plan_generation.application.safety_check import SafetyCheckResult
    from app.features.plan_generation.application.use_case import _build_escalation_reasons

    plan = SimpleNamespace(verdict=None, drug_interactions=[])
    reasons = _build_escalation_reasons(plan, SafetyCheckResult(), medication_omitted=["warfarin"])
    assert {"code": "medication_omitted", "count": 1} in reasons


# ── weight-bearing / surgeon-restriction gate (audit #10) ───────────────────


def _plan_wb(*weight_bearings: str):
    exs = [SimpleNamespace(weight_bearing=wb, laterality="") for wb in weight_bearings]
    return SimpleNamespace(phases=[SimpleNamespace(exercises=exs)])


@pytest.mark.parametrize(
    "case_text",
    [
        "Status post left THA, non-weight-bearing for 6 weeks.",  # en phrase
        "s/p ORIF, NWB left lower limb.",  # en abbrev
        "Состояние после остеосинтеза, без опоры на ногу.",  # ru
    ],
)
def test_case_nwb_plan_fwb_is_flagged(case_text):
    v = check_weight_bearing_fidelity(_plan_wb("FWB", "NWB"), case_text)
    assert v is not None
    assert v["case_level"] == "NWB"
    assert "FWB" in v["exceeding"]


def test_case_pwb_plan_wbat_is_flagged():
    v = check_weight_bearing_fidelity(_plan_wb("WBAT"), "Partial weight-bearing as per surgeon.")
    assert v is not None
    assert v["case_level"] == "PWB"
    assert v["exceeding"] == ["WBAT"]


def test_plan_within_ceiling_is_not_flagged():
    # Case NWB, plan stays NWB / bilateral n-a → no exceedance.
    assert check_weight_bearing_fidelity(_plan_wb("NWB", ""), "NWB for 6 weeks.") is None


def test_case_full_weight_bearing_never_flags():
    # FWB is the top of the scale; nothing can exceed it.
    assert (
        check_weight_bearing_fidelity(_plan_wb("FWB", "WBAT"), "Full weight-bearing allowed.")
        is None
    )


def test_no_weight_bearing_in_case_does_not_trip():
    # A non-orthopaedic case never states a WB status → gate never fires.
    assert check_weight_bearing_fidelity(_plan_wb("FWB"), "Type 2 diabetes, hypertension.") is None


def test_ambiguous_progression_two_levels_does_not_trip():
    # NWB then PWB names two levels → ambiguous which applies now → no flag.
    assert (
        check_weight_bearing_fidelity(_plan_wb("FWB"), "NWB for 2 weeks, then PWB, then WBAT.")
        is None
    )


def test_build_escalation_reasons_emits_weight_bearing_code():
    from app.features.plan_generation.application.safety_check import SafetyCheckResult
    from app.features.plan_generation.application.use_case import _build_escalation_reasons

    plan = SimpleNamespace(verdict=None, drug_interactions=[])
    reasons = _build_escalation_reasons(
        plan,
        SafetyCheckResult(),
        weight_bearing_exceeded={"case_level": "NWB", "exceeding": ["FWB"]},
    )
    assert {"code": "weight_bearing_exceeded"} in reasons


# ── monitoring-bounds sanity gate (audit #7) ────────────────────────────────


def _plan_monitoring(*bounds):
    # bounds: iterable of (parameter, min, max)
    mps = [
        SimpleNamespace(parameter=p, label=p, min_value=lo, max_value=hi) for (p, lo, hi) in bounds
    ]
    return SimpleNamespace(monitoring_parameters=mps)


def test_flipped_bound_is_flagged():
    plan = _plan_monitoring(("bp_systolic", 180, 90))
    assert check_monitoring_bounds(plan) == ["bp_systolic"]


def test_equal_bounds_are_flagged_degenerate():
    assert check_monitoring_bounds(_plan_monitoring(("heart_rate", 80, 80))) == ["heart_rate"]


def test_sane_bounds_not_flagged():
    assert check_monitoring_bounds(_plan_monitoring(("bp_systolic", 90, 160))) == []


def test_one_sided_bound_is_allowed():
    # Only a min (or only a max) is a legitimate one-sided alert, never flagged.
    assert check_monitoring_bounds(_plan_monitoring(("spo2", 92, None))) == []
    assert check_monitoring_bounds(_plan_monitoring(("temperature", None, 38.5))) == []


def test_monitoring_bounds_mixed():
    plan = _plan_monitoring(("bp_systolic", 90, 160), ("blood_glucose", 13, 5))
    assert check_monitoring_bounds(plan) == ["blood_glucose"]


def test_build_escalation_reasons_emits_monitoring_bounds_code():
    from app.features.plan_generation.application.safety_check import SafetyCheckResult
    from app.features.plan_generation.application.use_case import _build_escalation_reasons

    plan = SimpleNamespace(verdict=None, drug_interactions=[])
    reasons = _build_escalation_reasons(
        plan, SafetyCheckResult(), monitoring_bounds_invalid=["bp_systolic"]
    )
    assert {"code": "monitoring_bounds_invalid", "count": 1} in reasons


# ── discontinued-medication-still-scheduled gate (audit #5) ─────────────────


def test_case_discontinue_drug_still_scheduled_is_flagged():
    plan = _plan_with_meds("Warfarin 5 mg", "Amlodipine 5 mg")
    flagged = check_stopped_medication_scheduled(plan, "Discontinue warfarin due to bleeding risk.")
    assert flagged == ["warfarin"]


def test_case_hold_drug_still_scheduled_ru():
    plan = _plan_with_meds("Метформин 500 мг")
    flagged = check_stopped_medication_scheduled(
        plan, "Прекратить приём метформина перед операцией."
    )
    assert flagged == ["метформин"]


def test_plan_that_marks_the_drug_held_is_not_flagged():
    # The plan itself carries a STOP/HOLD marker → it is honouring the case.
    plan = SimpleNamespace(
        medication_schedule=[SimpleNamespace(medication="Warfarin", dose="HOLD", instructions="")]
    )
    assert check_stopped_medication_scheduled(plan, "Discontinue warfarin.") == []


def test_no_stop_verb_in_case_not_flagged():
    plan = _plan_with_meds("Warfarin 5 mg")
    assert check_stopped_medication_scheduled(plan, "Continue warfarin 5mg daily.") == []


def test_stop_verb_far_from_drug_not_flagged():
    # A stop verb about a different drug must not flag warfarin.
    plan = _plan_with_meds("Warfarin 5 mg")
    case = (
        "Continue warfarin as charted. Separately, discontinue the old antibiotic course entirely."
    )
    assert check_stopped_medication_scheduled(plan, case) == []


def test_build_escalation_reasons_emits_stopped_medication_code():
    from app.features.plan_generation.application.safety_check import SafetyCheckResult
    from app.features.plan_generation.application.use_case import _build_escalation_reasons

    plan = SimpleNamespace(verdict=None, drug_interactions=[])
    reasons = _build_escalation_reasons(
        plan, SafetyCheckResult(), stopped_medication_scheduled=["warfarin"]
    )
    assert {"code": "stopped_medication_scheduled", "count": 1} in reasons


# ── FAIL-CLOSED diagnosis-laterality guard (external-audit CRITICAL) ─────────
# Distinct from the flag-only per-exercise side check: this compares the case
# side against the plan's STATED DIAGNOSIS (plan_metadata.diagnosis_short) and is
# wired to BLOCK generation (raise LateralityMismatch), never merely escalate.


def _plan_diag(diagnosis_short: str):
    """A plan stub exposing only plan_metadata.diagnosis_short."""
    return SimpleNamespace(plan_metadata=SimpleNamespace(diagnosis_short=diagnosis_short))


def test_diagnosis_laterality_conflict_fires_on_opposite_side():
    from app.features.plan_generation.application.patient_fidelity import (
        check_diagnosis_laterality_conflict,
    )

    v = check_diagnosis_laterality_conflict(
        _plan_diag("Right total hip arthroplasty"), "Status post left total hip arthroplasty."
    )
    assert v == {"case_side": "left", "diagnosis_side": "right"}


def test_diagnosis_laterality_conflict_russian_flip_fires():
    from app.features.plan_generation.application.patient_fidelity import (
        check_diagnosis_laterality_conflict,
    )

    # Case LEFT (левого) vs plan diagnosis RIGHT (правого) → block.
    v = check_diagnosis_laterality_conflict(
        _plan_diag("Состояние после ТЭП правого ТБС"),
        "ТЭП левого тазобедренного сустава, двусторонний коксартроз, слева выраженнее",
    )
    assert v == {"case_side": "left", "diagnosis_side": "right"}


def test_diagnosis_laterality_matching_side_does_not_fire():
    """The external-audit case itself: left case + left diagnosis → NO block.

    Proves the guard does not fabricate a mismatch where none exists (the report's
    plan correctly stated the LEFT side, matching the LEFT-hip case).
    """
    from app.features.plan_generation.application.patient_fidelity import (
        check_diagnosis_laterality_conflict,
    )

    assert (
        check_diagnosis_laterality_conflict(
            _plan_diag("Состояние после ТЭП левого ТБС (6-е сутки)"),
            "ТЭП левого тазобедренного сустава, двусторонний коксартроз, слева выраженнее",
        )
        is None
    )
    assert (
        check_diagnosis_laterality_conflict(
            _plan_diag("Left total hip arthroplasty"), "Status post left THA."
        )
        is None
    )


def test_diagnosis_laterality_ambiguous_or_missing_never_fires():
    from app.features.plan_generation.application.patient_fidelity import (
        check_diagnosis_laterality_conflict,
    )

    # Bilateral / both-sides case → ambiguous → no block.
    assert (
        check_diagnosis_laterality_conflict(
            _plan_diag("Right THA"), "Bilateral coxarthrosis, left and right hips."
        )
        is None
    )
    # Diagnosis states no side → no block.
    assert (
        check_diagnosis_laterality_conflict(
            _plan_diag("Total hip arthroplasty, day 6"), "Left THA."
        )
        is None
    )
    # Case states no side → no block.
    assert (
        check_diagnosis_laterality_conflict(_plan_diag("Left THA"), "Chronic pain, no side stated.")
        is None
    )
    # Missing plan_metadata entirely → no crash, no block.
    assert check_diagnosis_laterality_conflict(SimpleNamespace(), "Left THA.") is None
