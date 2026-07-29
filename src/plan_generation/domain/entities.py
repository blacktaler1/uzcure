"""Domain entities for plan generation.

Pure data + invariants. No SQLAlchemy, no FastAPI, no Anthropic SDK.
All medical content originates from the LLM — these structures only enforce shape,
not clinical correctness (that is the doctor's role at approval time).

Internal-only fields (prefixed `_` or otherwise non-public):
  Exercise.exercise_id, Exercise.ai_suggested      — stripped from API response
  DrugInteraction.source                            — "deterministic" | "ai_extension"
                                                      stripped from API response
to_api_dict() helpers strip these fields recursively.
"""

from __future__ import annotations

import logging
from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Language
# ──────────────────────────────────────────────────────────────────────────


class PlanLanguage(str, Enum):
    EN = "en"
    RU = "ru"
    UZ = "uz"  # Uzbek — Latin script
    UZ_CYRILLIC = "uz-cyrl"  # Uzbek — Cyrillic script (Ўзбекча, кирилл алифбоси)

    @classmethod
    def from_str(cls, value: str | None) -> "PlanLanguage":
        if not value:
            return cls.UZ
        v = value.strip().lower()
        if v in {"en", "english", "eng"}:
            return cls.EN
        if v in {"ru", "russian", "rus"}:
            return cls.RU
        # Uzbek Cyrillic — check BEFORE the Latin-Uzbek branch so "uzbek_cyrillic"
        # etc. don't fall through to plain UZ.
        if v in {
            "uz-cyrl",
            "uz_cyrl",
            "uzc",
            "uzbek_cyrillic",
            "uzbek-cyrillic",
            "uzbek cyrillic",
            "kiril",
            "kirill",
            "cyrillic",
        }:
            return cls.UZ_CYRILLIC
        if v in {"uz", "uzbek", "uzb", "o'zbek", "ozbek", "uz-latn"}:
            return cls.UZ
        return cls.UZ

    @property
    def display_name(self) -> str:
        return {
            self.EN: "English",
            self.RU: "Russian",
            self.UZ: "Uzbek",
            self.UZ_CYRILLIC: "Uzbek (Cyrillic)",
        }[self]


# ──────────────────────────────────────────────────────────────────────────
# Input — command object (use case argument)
# ──────────────────────────────────────────────────────────────────────────


class GeneratePlanCommand(BaseModel):
    """Validated, immutable input for the use case. Built from API DTO."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doctor_id: int = Field(ge=1)
    clinic_id: int = Field(ge=1)
    patient_id: int | None = None
    patient_gender: Literal["male", "female"] | None = None
    is_pregnant: bool | None = None
    case_history_text: str = Field(min_length=20, max_length=50000)
    document_id: int | None = None
    output_language: PlanLanguage = PlanLanguage.UZ
    clinician_override: bool = False

    @field_validator("case_history_text")
    @classmethod
    def _strip(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 20:
            raise ValueError("case_history_text must be at least 20 chars after strip")
        return stripped


# ──────────────────────────────────────────────────────────────────────────
# Output — final plan structure (matches API contract verbatim)
# ──────────────────────────────────────────────────────────────────────────


class Exercise(BaseModel):
    """Exercise entry inside a phase.

    Public fields (API contract): name, sets, reps, duration_minutes,
    frequency, notes, has_video, video_url, origin.

    `origin` is the clean public signal of provenance — 'registry' (offered or
    resolved from the registry) or 'ai_suggested' (LLM-invented). The doctor must
    see which exercises are AI-invented; `exercise_id`/`ai_suggested` stay internal.
    `video_url` is public so a doctor-attached video URL survives the
    `GeneratedPlan.model_validate` round-trip in EditPlanUseCase.

    Internal fields (stripped from API, kept for audit + post-validation):
      exercise_id  — registry id, or None for AI-suggested
      ai_suggested — True when the LLM invented the exercise
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    sets: int | None = None
    reps: int | None = None
    duration_minutes: int | None = None
    frequency: str = ""
    # Structured, LANGUAGE-INDEPENDENT scheduling truth: how many days per week
    # this exercise is performed (1–7; 7 = every day). `frequency` above is a
    # localized display label only — it MUST NOT drive the calendar, because the
    # scheduler cannot reliably parse free text across uz/ru/en (a "3×/week" label
    # silently became a daily task). The scheduling compiler reads THIS field.
    days_per_week: int | None = Field(default=None, ge=1, le=7)
    notes: str = ""

    # ── Structured clinical prescription (P1) ──────────────────────────────
    # Dedicated slots so clinically-actionable detail is no longer buried in
    # free-text `notes`. All OPTIONAL: a plan that omits any of them is still
    # valid (never forces an LLM retry). String slots (not enums) because the
    # LLM owns the clinical value space; expected vocabularies are documented in
    # the output schema + prompt. Rendered on the doctor review screen.
    intensity: str = ""  # target intensity / Borg RPE, e.g. "moderate", "RPE 11-13"
    laterality: str = ""  # "left" | "right" | "bilateral" | "" (n/a)
    assistance_level: str = ""  # "independent" | "supervised" | "assisted" | "dependent"
    weight_bearing: str = ""  # "NWB" | "PWB" | "WBAT" | "FWB" | "" (n/a)
    hold_seconds: int | None = Field(default=None, ge=0, le=600)  # isometric hold per rep
    rest_seconds: int | None = Field(default=None, ge=0, le=600)  # rest between sets
    monitoring: str = ""  # what to watch during/after: HR, SpO2, symptoms, BP
    progression: str = ""  # how to advance this exercise
    regression: str = ""  # how to scale it back if not tolerated
    contraindications: str = ""  # when NOT to perform this exercise
    precautions: str = ""  # safety cautions while performing

    has_video: bool = False
    video_url: str = ""
    origin: Literal["registry", "ai_suggested"] = "registry"
    # Evidence provenance for a registry-backed exercise: the guideline/standard it
    # traces to (evidence_source, e.g. "ESSKA 2016") and its GRADE (evidence_level
    # A/B/C). PUBLIC — this is the plan's "cited" grounding for registry exercises
    # (which carry no PMIDs) and is shown to the doctor/patient. Server-filled from
    # the registry in resolve; the model never provides these.
    evidence_source: str = ""
    evidence_level: str = ""

    # Internal — stripped from public API response.
    exercise_id: str | None = None
    ai_suggested: bool = False
    # V4: PMIDs of retrieved evidence this exercise was grounded in (registry-miss
    # path only). INTERNAL — used for audit + post-generation PMID re-verification;
    # sources are NEVER shown to the doctor or patient.
    evidence_pmids: list[str] = Field(default_factory=list)


class Phase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    phase_number: int = Field(ge=1)
    phase_name: str
    duration: str
    goal: str
    # Optional, evidence-based phase criteria. The doctor sees these on the plan
    # review screen so the plan is actionable. Optional so a plan that omits them
    # is still valid (never forces an LLM retry); rendered only when present.
    #   entry_criteria       — what must be true to START this phase (readiness gate)
    #   progression_criteria — what must be true to ADVANCE out of this phase
    entry_criteria: str = ""
    progression_criteria: str = ""
    exercises: list[Exercise] = Field(min_length=1)

    @field_validator("exercises", mode="before")
    @classmethod
    def _truncate_exercises(cls, v: object) -> object:
        if isinstance(v, list) and len(v) > 6:
            _logger.warning("Phase received %d exercises; truncating to 6.", len(v))
            return v[:6]
        return v


class PlanMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    patient_name: str
    patient_id: str
    diagnosis_short: str
    start_date: date
    total_duration_weeks: int | None = Field(default=None, ge=1, le=52)
    total_duration_days: int | None = Field(default=None, ge=1)
    # OPTIONAL weekly rest/lighter day. Was required — which forced a fixed weekly
    # "reward day" onto EVERY plan, including emergency abstentions (a "go to the ER"
    # plan carrying "Reward day (rest): Sunday" is clinically absurd). A rest day is
    # case-dependent and should follow fatigue/participation, not a fixed calendar
    # slot, and it never belongs on a withheld emergency plan. Empty default → a plan
    # may omit it; the renderer already shows the row only when it is present.
    reward_day: str = ""
    post_discharge_cap_weeks: int = Field(ge=0, le=52)
    post_discharge_note: str


class MedicationScheduleEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    time: str
    medication: str
    dose: str
    instructions: str
    duration: str
    frequency: str = ""
    dose_times: list[str] = Field(default_factory=list)
    notes: str = ""
    # V4: internal-only evidence grounding (PMIDs the LLM may cite for this
    # medication). Gated by _enforce_pmid_gate; stripped from the public API
    # in to_api_dict, exactly like Exercise.evidence_pmids.
    evidence_pmids: list[str] = Field(default_factory=list)


class DrugInteraction(BaseModel):
    """Drug interaction surfaced to the patient/doctor.

    Public fields: title, description, severity.
    Internal `source` distinguishes deterministic vs AI-extension origin —
    stripped from API but written to job audit metadata for admin review.
    """

    model_config = ConfigDict(extra="ignore")

    title: str
    description: str
    severity: Literal["Caution", "OK", "Block"]

    # Internal provenance — "deterministic" (from medication_rules) or
    # "ai_extension" (LLM-supplied). Public response strips this.
    source: Literal["deterministic", "ai_extension"] = "ai_extension"


class Supplement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    reason: str
    frequency: str
    duration: str
    # V4: internal-only evidence grounding (PMIDs the LLM may cite for this
    # supplement recommendation). Gated by _enforce_pmid_gate, re-verified at
    # NCBI, and stripped from the public API in to_api_dict — exactly like
    # Exercise.evidence_pmids and MedicationScheduleEntry.evidence_pmids.
    evidence_pmids: list[str] = Field(default_factory=list)


class FoodCategory(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: str
    examples: str
    purpose: str


class FoodAvoid(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category: str
    reason: str


class DietPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tailored_for: list[str] = Field(default_factory=list)
    recommended_foods: list[FoodCategory] = Field(default_factory=list)
    foods_to_avoid: list[FoodAvoid] = Field(default_factory=list)
    # Supplement recommendations live INSIDE the diet plan: nutrition and
    # supplementation are one clinical surface (they substitute for each other —
    # e.g. "oily fish" vs "fish-oil capsule"), so generating them together saves
    # tokens/latency and avoids a redundant top-level block. NOTE: this is the
    # LLM's supplement *recommendation* output only — the deterministic
    # drug↔supplement *interaction* safety layer (SupplementCatalogue,
    # supplementation here are plan-owned, produced by the LLM.
    supplements: list[Supplement] = Field(default_factory=list)
    additional_recommendations: list[str] = Field(default_factory=list)


class SafetyAlert(BaseModel):
    model_config = ConfigDict(extra="ignore")

    alert: str
    detail: str


class Actions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    approve_button: str
    edit_button: str
    reject_button: str


class Verdict(str, Enum):
    """Triage outcome. Single signal driving how urgently the doctor must review.
    The rehab plan body (phases, safety_alerts, pre_clearance_criteria,
    emergency_stop_criteria, emergency_action) carries the actual warnings —
    verdict is NOT a separate data block, just a severity tag.

    proceed — patient stable; plan immediately actionable
    caution — patient medically complex or has stacked risk factors; doctor
              should review specific clamps/contraindications before delivery

    Abstention / fail-closed states — NO plan body is produced (phases empty).
    The correct medical output is to withhold a plan and tell the doctor why, so
    the system never fabricates a rehabilitation plan on missing data, an
    exercise contraindication, or unavailable evidence:
    insufficient_information — the case lacks data required to prescribe safely
                               (e.g. weight-bearing status / post-op precautions
                               unclear, no functional baseline).
    contraindicated          — an active condition makes rehab generation unsafe
                               until clearance (e.g. unstable cardiac state,
                               suspected DVT/PE, unstable fracture).
    evidence_unavailable     — required grounding evidence could not be retrieved,
                               so an evidence-governed plan cannot be emitted.
    """

    PROCEED = "proceed"
    CAUTION = "caution"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    CONTRAINDICATED = "contraindicated"
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"

    @property
    def abstains(self) -> bool:
        """True for the fail-closed states that must NOT carry a plan body."""
        return self in {
            Verdict.INSUFFICIENT_INFORMATION,
            Verdict.CONTRAINDICATED,
            Verdict.EVIDENCE_UNAVAILABLE,
        }

    @classmethod
    def _missing_(cls, value):  # legacy DB rows may still carry escalate/monitoring_only
        if isinstance(value, str) and value.lower() in {"escalate", "monitoring_only"}:
            return cls.CAUTION
        return None


# Fail-closed review scaffolding applied ONLY to a caution plan when the model left
# the field blank. These are GENERIC, PROCEDURAL review gates -- not clinical
# knowledge: no drug, dose, diagnosis, or condition-specific content. The LLM stays
# the sole clinical authority, so anything it supplies is preserved untouched and a
# proceed plan is never modified. Keyed by the plan's output_language (en/ru/uz/
# uz-cyrl) so a Russian/English caution plan never shows Uzbek review gates; falls
# back to English for an unknown/blank language.
_CAUTION_POST_DISCHARGE_NOTE: dict[str, str] = {
    "en": (
        "Continue rehabilitation under physician supervision; "
        "attend all scheduled follow-up visits."
    ),
    "ru": (
        "Продолжайте реабилитацию под наблюдением врача; "
        "посещайте все запланированные контрольные осмотры."
    ),
    "uz": (
        "Reabilitatsiyani shifokor nazorati ostida davom ettiring; "
        "rejalashtirilgan nazorat ko'riklariga rioya qiling."
    ),
    "uz-cyrl": (
        "Реабилитацияни шифокор назорати остида давом эттиринг; "
        "режалаштирилган назорат кўрикларига риоя қилинг."
    ),
}
_CAUTION_PRE_CLEARANCE_FLOOR: dict[str, tuple[str, ...]] = {
    "en": (
        "A physician must confirm a final review before this plan is delivered.",
        "Vital-sign stability must be confirmed.",
        "The absence of new warning signs must be checked.",
    ),
    "ru": (
        "Врач должен подтвердить окончательный осмотр перед выдачей этого плана.",
        "Должна быть подтверждена стабильность жизненных показателей.",
        "Должно быть проверено отсутствие новых тревожных признаков.",
    ),
    "uz": (
        "Shifokor reja yetkazilishidan oldin yakuniy ko'rikni tasdiqlasin.",
        "Hayotiy ko'rsatkichlar barqarorligi tasdiqlansin.",
        "Yangi ogohlantiruvchi belgilar yo'qligi tekshirilsin.",
    ),
    "uz-cyrl": (
        "Шифокор режа етказилишидан олдин якуний кўрикни тасдиқласин.",
        "Ҳаётий кўрсаткичлар барқарорлиги тасдиқлансин.",
        "Янги огоҳлантирувчи белгилар йўқлиги текширилсин.",
    ),
}

# Localized label for a safety alert PROMOTED from an emergency-stop criterion when
# the model returned no explicit safety_alerts. Only the heading is localized — the
# criterion text itself is the model's own (already in the plan language). Keyed by
# GeneratedPlan.output_language (en/ru/uz/uz-cyrl); falls back to English for an
# unknown/blank language, mirroring the localized-label pattern in
# application/safety_check.py so a non-Uzbek plan never shows an Uzbek heading.
_EMERGENCY_STOP_ALERT_LABEL: dict[str, str] = {
    "en": "Critical stop criterion",
    "ru": "Критический критерий остановки",
    "uz": "Muhim to'xtatish mezoni",
    "uz-cyrl": "Муҳим тўхтатиш мезони",
}


class MonitoringParameter(BaseModel):
    """A physiological parameter the plan says to monitor for THIS patient, with
    the bounds that make a reading actionable.

    LLM-DERIVED from the patient's own conditions — a diabetic's plan carries a
    ``blood_glucose`` parameter, a non-diabetic's does not — NOT a hardcoded
    per-condition table. Runtime monitoring is therefore condition-gated *by
    construction*: a parameter absent from the plan is never evaluated, so an
    unrelated patient (e.g. no diabetes) is never alarmed about it. This mirrors
    the plan-derived pain threshold that safety_monitor already consumes, and
    keeps clinical thresholds out of application code per the universality
    mandate (the ONLY hardcoded clinical content is the exercise registry).

    ``parameter`` is a canonical measurement key (a field NAME, not clinical
    knowledge) so the runtime can map a submitted vital to its bound; known keys:
    blood_glucose | bp_systolic | bp_diastolic | heart_rate | spo2 |
    temperature | weight | respiratory_rate. An unknown key is still shown to
    the doctor/patient but not auto-evaluated.
    """

    model_config = ConfigDict(extra="ignore")

    parameter: str
    label: str
    reason: str
    unit: str = ""
    min_value: float | None = None
    max_value: float | None = None
    frequency: str = ""
    # True when this parameter's safe thresholds are PATIENT-SPECIFIC and must be
    # set/confirmed by the clinician before the alert is activated — e.g. a glucose
    # band for an insulin-treated patient with a personal hypo threshold, a cardiac
    # target under a rate-limiting drug, or an orthostatic BP range. The LLM must NOT
    # present an invented cut-off for these as if it were an active, patient-ready
    # alarm; it emits its values as a SUGGESTION and flags this so the doctor
    # configures the real threshold. min_value/max_value may still be present as the
    # suggested starting point. PUBLIC — the renderer shows a "clinician
    # configuration required before activation" badge.
    clinician_config_required: bool = False
    # Internal-only evidence grounding, stripped from the public API in
    # to_api_dict exactly like Exercise/MedicationScheduleEntry/Supplement.
    evidence_pmids: list[str] = Field(default_factory=list)


class Referral(BaseModel):
    """A multidisciplinary referral the case warrants (LLM-derived).

    Universal across every medical field: the model names the discipline THIS
    patient needs — speech-language therapy, occupational therapy, dietitian,
    orthotics/prosthetics (AFO/FES), clinical psychology, social work, continence/
    urology, wound care, etc. — never a hardcoded per-condition set. Optional: the
    list is empty when the case needs no discipline beyond the prescribing plan, so
    a simple single-domain case never carries a spurious referral.
    """

    model_config = ConfigDict(extra="ignore")

    discipline: str  # e.g. "Speech-language therapy", "Occupational therapy"
    reason: str  # why this referral is indicated for THIS patient
    urgency: str = ""  # "" | "routine" | "soon" | "urgent"


class Assessment(BaseModel):
    """A clinical assessment / screen the plan recommends (LLM-derived).

    The domain-agnostic slot for the evaluations a complex case needs before or
    during rehab — swallowing/dysphagia screen, cognitive-perceptual assessment,
    shoulder examination (subluxation / rotator cuff / CRPS), bladder assessment,
    mood / post-event depression screen, home / discharge-environment assessment,
    etc. Optional; empty when nothing beyond the standard exam is warranted.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    reason: str
    timing: str = ""  # "" | "baseline" | "before progression" | "ongoing"


class OutcomeMeasure(BaseModel):
    """A validated outcome measure to baseline and re-measure (LLM-derived).

    Condition-appropriate and universal: Barthel Index / FIM, Berg Balance Scale,
    Functional Ambulation Category, 10-Metre & 6-Minute Walk Tests, Fugl-Meyer,
    NIHSS / modified Rankin, a validated pain or language measure, etc. Optional;
    empty when a simple case needs no formal instrument. Progression should key off
    these measures rather than calendar weeks alone.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    purpose: str = ""  # what it tracks — function, balance, gait, motor, cognition
    baseline_timing: str = ""  # when to measure first (e.g. "baseline", "week 2")
    target: str = ""  # goal / meaningful-change threshold if applicable


class InterventionProgramme(BaseModel):
    """A STRUCTURED non-exercise treatment programme for ONE impairment domain
    (LLM-derived). This is the root fix for the "multidisciplinary in name,
    physiotherapy in prescription" gap: `referrals`/`assessments` are POINTERS
    ("see SLP"), plain-string `therapy_schedule`/`care_coordination` are not
    prescriptions, and `phases[].exercises[]` — the only rigorous prescription
    container — is physiotherapy-shaped. Every non-exercise domain a complex case
    presents (communication/aphasia/dysarthria, swallowing/dysphagia, self-care ADL
    retraining, cognition, mood/motivation, continence, caregiver training,
    pressure-injury prevention, community reintegration) needs the SAME rigor as an
    exercise: concrete ACTIVITIES, a FREQUENCY, its OWN advance/hold criteria
    (per-intervention, never a single global score), and a bound OUTCOME MEASURE —
    not merely a referral. Optional/empty for a simple single-domain case; the model
    emits one programme per impairing domain the case documents, alongside (not
    instead of) `referrals` and `phases`.
    """

    model_config = ConfigDict(extra="ignore")

    domain: str  # e.g. "swallowing", "communication", "self-care ADL", "cognition", "mood", "continence", "caregiver", "skin integrity"
    discipline: str  # who delivers it — e.g. "Speech-language therapy", "Occupational therapy", "Psychology", "Nursing", "Dietetics"
    goal: str = ""
    activities: list[str] = Field(
        default_factory=list
    )  # the concrete prescribed intervention activities
    frequency: str = ""  # e.g. "3x/week, 30 min"; capacity-based, not calendar-fixed
    advance_when: str = (
        ""  # criteria to progress THIS programme (per-intervention, not a global gate)
    )
    hold_or_regress_when: str = ""  # criteria to hold / step back
    outcome_measure: str = ""  # the validated instrument that tracks THIS domain
    precautions: str = ""


def _norm_alert_text(s: str) -> str:
    """Normalize a safety-alert string for duplicate detection: case-fold, collapse
    internal whitespace, and strip surrounding whitespace + trailing punctuation. So
    "Monitor for DVT" and "monitor for dvt." collapse to the same key, but genuinely
    distinct wording stays distinct."""
    import re

    return re.sub(r"\s+", " ", (s or "").strip().casefold()).strip(" .,;:!?-–—")


class GeneratedPlan(BaseModel):
    """Final plan delivered to API. Shape is the public contract."""

    # Tolerate extra LLM keys at every level so a single unsolicited field never
    # costs a full retry. Required-field and enum constraints still enforce shape.
    model_config = ConfigDict(extra="ignore")

    plan_metadata: PlanMetadata
    verdict: Verdict = Verdict.PROCEED
    decision: Literal["proceed", "caution"] = "proceed"
    # Empty ONLY when the verdict abstains (fail-closed). A non-abstaining plan
    # MUST carry ≥1 phase — enforced in `_enforce_fail_closed`, not as a field
    # constraint, so an abstention with no phases is a valid shape rather than a
    # hard validation error that would force the LLM to fabricate a plan.
    phases: list[Phase] = Field(default_factory=list, max_length=8)
    # Doctor-facing clinical reason the plan was withheld. Required (non-empty)
    # for an abstaining verdict; empty for proceed/caution.
    abstention_reason: str = ""
    # The language the plan was generated in (en/ru/uz/uz-cyrl). Carried through
    # to_api_dict so the review/patient UI renders the plan's STRUCTURAL labels
    # (Exercise, Dosage, Side, ...) in the plan's own language — not the viewer's
    # UI language — so a Russian plan reads coherently even under an English UI.
    output_language: str = ""
    medication_schedule: list[MedicationScheduleEntry] = Field(default_factory=list)
    drug_interactions: list[DrugInteraction] = Field(default_factory=list)
    diet_plan: DietPlan
    safety_alerts: list[SafetyAlert] = Field(default_factory=list)
    # Per-patient physiological parameters to monitor (LLM-derived from the
    # patient's conditions; empty when nothing warrants monitoring). Optional so
    # a thin plan never fails validation. Runtime alerting reads these bounds and
    # is condition-gated by their presence — see MonitoringParameter.
    monitoring_parameters: list[MonitoringParameter] = Field(default_factory=list)
    pre_clearance_criteria: list[str] = Field(default_factory=list)
    emergency_stop_criteria: list[str] = Field(default_factory=list)
    emergency_action: str
    actions: Actions

    # ── Multidisciplinary care domains (universal, LLM-derived) ─────────────
    # Structured slots so a complex case's coordinated-care needs are no longer
    # lost for want of a field (the prompt asked for referrals / outcome measures
    # but there was nowhere to put them). All OPTIONAL and condition-agnostic:
    # the model populates them from the case exactly like phases/medications, so a
    # simple single-domain plan carries empty lists while a multi-morbid case
    # (e.g. stroke needing SLT/OT/swallowing/cognition/outcome-measures/caregiver)
    # carries them. No hardcoded per-condition content — the universality mandate
    # holds. Rendered on the doctor-review + patient plan; never gate generation.
    referrals: list[Referral] = Field(default_factory=list)
    assessments: list[Assessment] = Field(default_factory=list)
    outcome_measures: list[OutcomeMeasure] = Field(default_factory=list)
    # STRUCTURED non-exercise treatment programmes — one per impairing domain the
    # case presents (SLT/OT/psychology/nursing/etc.), each with concrete activities,
    # frequency, its own advance/hold criteria and a bound outcome measure. Turns a
    # multidisciplinary plan from "a pile of referral pointers" into real, trackable
    # prescriptions with the same rigor as `phases[].exercises[]`. Optional; empty
    # for a simple single-domain case. Never gates generation.
    intervention_programmes: list[InterventionProgramme] = Field(
        default_factory=list, max_length=12
    )
    # Coordinated multidisciplinary therapy DOSE — the daily/weekly time allocation
    # and session structure across the disciplines the case needs (e.g. how much
    # PT / OT / SLT per day, split into blocks for a fatigue-prone patient). The
    # per-exercise sets/reps do not express a coordinated therapy intensity, so a
    # complex case's dose was previously invisible. Free-text lines, LLM-derived,
    # universal; empty for a simple single-discipline case.
    therapy_schedule: list[str] = Field(default_factory=list)
    # Caregiver capacity, equipment / assistive-device needs, and home &
    # discharge-planning items the case warrants (free-text lines, LLM-derived).
    care_coordination: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_verdict_decision(self) -> "GeneratedPlan":
        """Reconcile the two triage representations toward the SAFER state.

        `verdict` (the Verdict enum) and `decision` (the legacy string the UI
        branches on) encode the same triage signal. A caller or the LLM may
        populate EITHER one. This method previously copied `verdict` onto
        `decision` unconditionally, so a `caution` arriving only via `decision`
        (with `verdict` left at its PROCEED default) was silently overwritten back
        to `proceed` -- hiding the caution from the doctor-review UI. Caution is a
        one-way ratchet: if EITHER field is caution, BOTH become caution, so the
        signal can never be downgraded regardless of which field carried it.

        Order matters: this runs before `_normalize_caution`, so a decision-only
        caution also lifts `verdict` to CAUTION in time to trigger the
        caution-driven safety-alert promotion below.
        """
        if self.verdict.abstains:
            # Abstention carries the real state in `verdict`; the legacy 2-value
            # `decision` the review UI branches on maps to "caution" so an
            # abstaining plan can never read as an actionable "proceed".
            self.decision = "caution"  # type: ignore[assignment]
        elif self.verdict == Verdict.CAUTION or self.decision == "caution":
            self.verdict = Verdict.CAUTION
            self.decision = "caution"  # type: ignore[assignment]
        else:
            self.decision = self.verdict.value  # type: ignore[assignment]
        return self

    @model_validator(mode="after")
    def _enforce_fail_closed(self) -> "GeneratedPlan":
        """Guarantee the fail-closed contract:
        * an abstaining verdict MUST give a clinical reason and carries NO plan
          body (phases are cleared — the doctor acts on the reason, they cannot
          approve/deliver a non-existent plan);
        * a non-abstaining verdict MUST carry ≥1 phase (the rehab plan is the
          product). This replaces the old `phases` min_length field constraint.
        """
        if self.verdict.abstains:
            if not (self.abstention_reason or "").strip():
                raise ValueError(
                    f"verdict '{self.verdict.value}' abstains and requires a "
                    f"non-empty abstention_reason"
                )
            self.phases = []
        elif not self.phases:
            raise ValueError("a non-abstaining plan must contain at least one phase")
        return self

    @model_validator(mode="after")
    def _normalize_safety_alerts(self) -> "GeneratedPlan":
        if self.safety_alerts:
            return self
        promoted: list[SafetyAlert] = [
            SafetyAlert(alert=di.title, detail=di.description)
            for di in self.drug_interactions
            if di.severity == "Block"
        ]
        if not promoted and self.emergency_stop_criteria:
            label = _EMERGENCY_STOP_ALERT_LABEL.get(
                str(self.output_language), _EMERGENCY_STOP_ALERT_LABEL["en"]
            )
            promoted = [
                SafetyAlert(alert=label, detail=c) for c in self.emergency_stop_criteria[:3]
            ]
        if promoted:
            self.safety_alerts = promoted
        return self

    @model_validator(mode="after")
    def _normalize_caution(self) -> "GeneratedPlan":
        """Caution backstop -- defense in depth for a doctor-review plan.

        Runs AFTER ``_normalize_verdict_decision`` (so a decision-only caution has
        already lifted ``verdict`` to CAUTION) and AFTER ``_normalize_safety_alerts``
        (so Block / emergency promotion is already applied). For a caution plan ONLY:

          * promote EVERY drug interaction (any severity -- Block, Caution, OK) into
            ``safety_alerts``, deduped by title against what is already there and
            within the interaction list itself. A caution plan must never reach the
            reviewing doctor with the interaction notes hidden, even the low-severity
            ones, because caution means "the doctor should re-check the specifics".
          * guarantee the review scaffolding is non-empty: a post-discharge note and
            at least three pre-clearance gates. These are generic procedural floors
            (see ``_CAUTION_*`` constants) applied only when the model left the field
            blank; anything the model supplied is preserved untouched.

        A proceed plan is returned unchanged.
        """
        if self.verdict != Verdict.CAUTION:
            return self

        # Promote interactions of ANY severity, deduped by title.
        seen = {sa.alert for sa in self.safety_alerts}
        extra: list[SafetyAlert] = []
        for di in self.drug_interactions:
            if di.title in seen:
                continue
            seen.add(di.title)
            extra.append(SafetyAlert(alert=di.title, detail=di.description))
        if extra:
            self.safety_alerts = list(self.safety_alerts) + extra

        # Fail-closed review floors (generic / procedural -- never clinical content),
        # localized to the plan's own language so a RU/EN caution plan never shows
        # Uzbek review gates. output_language may still be "" here if this validator
        # runs before the server stamps the language — the caller injects it into
        # model_validate so the correct language is available; en is the fallback.
        _lang = str(self.output_language) or "en"
        if not self.plan_metadata.post_discharge_note.strip():
            self.plan_metadata.post_discharge_note = _CAUTION_POST_DISCHARGE_NOTE.get(
                _lang, _CAUTION_POST_DISCHARGE_NOTE["en"]
            )
        if not self.pre_clearance_criteria:
            self.pre_clearance_criteria = list(
                _CAUTION_PRE_CLEARANCE_FLOOR.get(_lang, _CAUTION_PRE_CLEARANCE_FLOOR["en"])
            )

        return self

    @model_validator(mode="after")
    def _dedupe_safety_alerts(self) -> "GeneratedPlan":
        """Collapse duplicate safety alerts, keeping the first occurrence.

        Runs LAST — after every promotion (`_normalize_safety_alerts` Block/emergency,
        `_normalize_caution` drug-interaction promotion) — so the doctor never sees the
        same warning repeated: whether the LLM emitted it twice across phases, or a
        promotion restated an alert already present. Dedup key is the normalized
        (alert, detail) pair, so an exact repeat and a case/whitespace/trailing-
        punctuation variant collapse, while a genuinely distinct alert (different
        title OR different detail) is always preserved. Applies to EVERY verdict —
        the earlier promotion validators only deduped within their own additions."""
        if len(self.safety_alerts) < 2:
            return self
        seen: set[tuple[str, str]] = set()
        deduped: list[SafetyAlert] = []
        for sa in self.safety_alerts:
            key = (_norm_alert_text(sa.alert), _norm_alert_text(sa.detail))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(sa)
        if len(deduped) != len(self.safety_alerts):
            self.safety_alerts = deduped
        return self

    def to_api_dict(self) -> dict[str, Any]:
        """Public API representation — internal fields stripped."""
        raw = self.model_dump(mode="json")
        for phase in raw.get("phases", []):
            for ex in phase.get("exercises", []):
                # `origin` and `video_url` are KEPT (public provenance + doctor-
                # attached video). Only the raw registry id + the boolean flag
                # behind `origin` are stripped.
                ex.pop("exercise_id", None)
                ex.pop("ai_suggested", None)
                # V4: evidence sources are INTERNAL ONLY — never shown to the
                # doctor or patient. Stripped here like the other internal fields.
                ex.pop("evidence_pmids", None)
        # V4: medication-schedule evidence PMIDs are internal-only too — strip
        # them from the public API exactly like the per-exercise evidence above.
        for med in raw.get("medication_schedule", []):
            med.pop("evidence_pmids", None)
        # V4: monitoring-parameter evidence PMIDs are internal-only — strip them
        # from the public API exactly like the per-exercise/medication evidence.
        for mp in raw.get("monitoring_parameters", []):
            mp.pop("evidence_pmids", None)
        # V4: supplement evidence PMIDs (now nested under diet_plan) are likewise
        # internal-only — strip them from the public API.
        _diet = raw.get("diet_plan")
        if isinstance(_diet, dict):
            for supp in _diet.get("supplements", []):
                supp.pop("evidence_pmids", None)
        # `source` is intentionally KEPT (provenance, not PHI): the doctor sees
        # whether an interaction came from the curated deterministic DB
        # ("deterministic") or the LLM ("ai_extension"), to render the trusted
        # tier. The audit_meta counts remain available for ops.
        return raw

    def audit_meta(self) -> dict[str, Any]:
        """Audit-only view: counts of internal flags for the job result __meta block."""
        ai_sugg = sum(1 for ph in self.phases for ex in ph.exercises if ex.ai_suggested)
        det_di = sum(1 for di in self.drug_interactions if di.source == "deterministic")
        ai_di = sum(1 for di in self.drug_interactions if di.source == "ai_extension")
        return {
            "ai_suggested_exercise_count": ai_sugg,
            "deterministic_drug_interactions": det_di,
            "ai_extension_drug_interactions": ai_di,
            "verdict": self.verdict.value,
        }

    def exercise_provenance(self) -> list[dict[str, Any]]:
        """Server-only per-exercise provenance that MUST survive generation ->
        approval -> delivery. `to_api_dict()` strips exercise_id/evidence_pmids from
        the public phases (the client uses `origin` as the provenance signal), but
        the plan writer needs the registry exercise_id to persist provenance and
        link registry videos. This list is carried in the job result's `__meta`
        (internal audit block) and read by the writer, keyed by (phase_number,
        index) so it can be matched back to each exercise deterministically."""
        out: list[dict[str, Any]] = []
        for ph in self.phases:
            for i, ex in enumerate(ph.exercises):
                out.append(
                    {
                        "phase_number": ph.phase_number,
                        "index": i,
                        # `name` makes the sidecar EDIT-STABLE: the plan writer
                        # matches provenance by (phase_number, name), which survives
                        # a composer delete/reorder (indices shift, the name travels
                        # with the exercise). `index` is kept as a legacy fallback.
                        "name": ex.name,
                        "exercise_id": ex.exercise_id,
                        "origin": ex.origin,
                        "ai_suggested": ex.ai_suggested,
                        "evidence_pmids": list(ex.evidence_pmids),
                        "evidence_source": ex.evidence_source or None,
                        "evidence_level": ex.evidence_level or None,
                    }
                )
        return out

    @property
    def is_actionable(self) -> bool:
        """True when the plan can be approved + delivered as a rehab plan."""
        return self.verdict in (Verdict.PROCEED, Verdict.CAUTION)
