"""Post-LLM safety check.

Pure logic. Mutates `GeneratedPlan` exercise dosing in place to keep doses
inside the registry envelope, scans for diagnosis ↔ exercise contraindication
overlap, and returns a `SafetyCheckResult` summarising every clamp / warning.

Replaces the legacy `verification_fortress.py` (Stage 2 + Stage 3 only).

AI-invented exercises (no registry id) are NOT skipped: they get generic
absolute dose ceilings + a diagnosis-term scan of their name/notes so they can
no longer bypass all post-LLM safety (C3). Exercise-id hallucination *removal*
(the ai_suggested flag + null id) still happens in use_case `_resolve_exercises`.

What intentionally does NOT live here:
  - phase RPE check, evidence ledger, scoring — out of scope for this port.
  - WEB_EVIDENCE_ / SAFE-FB- prefixes — new pipeline does not generate them.

Decision: warn, never block. Plan always returns to the doctor for review;
this layer surfaces clamp / contraindication metadata for UI badges and audit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.features.plan_generation.application.ports import (
    PatientSnapshot,
    RegistryExercise,
)
from app.features.plan_generation.domain import Exercise, GeneratedPlan, SafetyAlert, Verdict

logger = logging.getLogger(__name__)


# Generic absolute dose ceilings for AI-invented exercises with no registry
# envelope. These are NOT clinical prescriptions — they are sane upper/lower
# bounds that catch egregious LLM dosing (e.g. "99 reps") so a doctor reviewing
# the draft is not handed an obviously unsafe number. The doctor sets the real
# dose at approval. Out-of-range values are clamped to the nearest bound.
_INVENTED_SETS_MIN = 1
_INVENTED_SETS_MAX = 5
_INVENTED_REPS_MIN = 1
_INVENTED_REPS_MAX = 30
_INVENTED_DURATION_MIN = 1
_INVENTED_DURATION_MAX = 60


# ──────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class DoseClamp:
    exercise_id: str
    changes: list[str]


@dataclass
class DoseSynthesis:
    """Schema mismatch fix. Not a safety violation — the LLM gave the wrong
    dosing schema (e.g. sets/reps for a duration-based exercise) and we
    derived a sensible default. Logged for observability, doctor review NOT
    forced solely on these.
    """

    exercise_id: str
    reason: str
    detail: str


@dataclass
class ContraindicationHit:
    exercise_id: str
    diagnosis_term: str
    contraindication: str


@dataclass
class AllergyHit:
    """A patient allergen matched against something the plan prescribes.

    `where` is a PHI-free location label ("exercise" | "supplement" | "food" |
    "medication"); `matched` is the plan label that triggered the hit (clinical
    text, not patient PHI). Flag-only — never strips the plan.
    """

    allergen: str
    where: str
    matched: str


@dataclass
class SafetyCheckResult:
    clamps: list[DoseClamp] = field(default_factory=list)
    synthesis: list[DoseSynthesis] = field(default_factory=list)
    contraindications: list[ContraindicationHit] = field(default_factory=list)
    allergy_hits: list[AllergyHit] = field(default_factory=list)

    @property
    def safety_clamps_count(self) -> int:
        return len(self.clamps)

    @property
    def safety_warnings_count(self) -> int:
        return len(self.contraindications) + len(self.allergy_hits)

    def to_meta(self) -> dict[str, Any]:
        return {
            "clamps": [{"exercise_id": c.exercise_id, "changes": c.changes} for c in self.clamps],
            "synthesis": [
                {"exercise_id": s.exercise_id, "reason": s.reason, "detail": s.detail}
                for s in self.synthesis
            ],
            "contraindications": [
                {
                    "exercise_id": h.exercise_id,
                    "diagnosis_term": h.diagnosis_term,
                    "contraindication": h.contraindication,
                }
                for h in self.contraindications
            ],
            "allergy_hits": [
                {"allergen": a.allergen, "where": a.where, "matched": a.matched}
                for a in self.allergy_hits
            ],
        }


# ──────────────────────────────────────────────────────────────────────────
# Public entry
# ──────────────────────────────────────────────────────────────────────────


def run_safety_check(
    plan: GeneratedPlan,
    *,
    pool_by_id: dict[str, RegistryExercise],
    patient: PatientSnapshot | None,
    case_text: str | None = None,
) -> SafetyCheckResult:
    """Mutate `plan` in place; return a summary of changes.

    ``case_text`` (the free-text case) is scanned for explicitly-stated allergens
    and unioned with the structured record, so the allergen cross-check honours a
    documented allergy regardless of where it was recorded.
    """
    result = SafetyCheckResult()
    diag_terms = _diagnosis_terms(patient)

    for phase in plan.phases:
        for exercise in phase.exercises:
            if exercise.exercise_id is None:
                # AI-invented exercise — no registry envelope. Apply generic
                # absolute ceilings + scan the invented name/notes against the
                # patient's diagnosis terms so an invented exercise can no longer
                # bypass ALL post-LLM safety (closes C3). Never strip — flag only.
                _clamp_invented(exercise, result)
                _scan_invented_contraindications(exercise, diag_terms, result)
                continue
            registry = pool_by_id.get(exercise.exercise_id)
            if registry is None:
                continue

            _clamp_dose(exercise, registry, result)
            _scan_contraindications(exercise, registry, diag_terms, result)

    _scan_allergies(plan, merge_allergens(patient, case_text), result)

    if result.clamps:
        logger.info("safety_check_clamped count=%d", len(result.clamps))
    if result.contraindications:
        logger.info("safety_check_contra_hits count=%d", len(result.contraindications))
    if result.allergy_hits:
        # PHI: count only — allergen / drug / food names are PHI (security.md).
        logger.info("safety_check_allergy_hits count=%d", len(result.allergy_hits))

    return result


# ──────────────────────────────────────────────────────────────────────────
# Stage: dose envelope
# ──────────────────────────────────────────────────────────────────────────


def _clamp_dose(
    exercise: Exercise,
    registry: RegistryExercise,
    result: SafetyCheckResult,
) -> None:
    dr = registry.dosage_ranges
    if not isinstance(dr, dict):
        return

    is_non_rep = dr.get("reps_max") == 1 and dr.get("reps_min") == 1
    has_duration_envelope = is_non_rep and ("duration_max" in dr or "duration_min" in dr)

    if has_duration_envelope:
        _handle_duration_based(exercise, dr, result)
        return

    if is_non_rep:
        _handle_education(exercise, dr, result)
        return

    _handle_rep_based(exercise, dr, result)


def _handle_rep_based(
    exercise: Exercise,
    dr: dict[str, Any],
    result: SafetyCheckResult,
) -> None:
    changes: list[str] = []

    sets_min, sets_max = _envelope_pair(dr, "sets_min", "sets_max")
    if sets_min is not None and sets_max is not None and exercise.sets:
        s = int(exercise.sets)
        if s < sets_min:
            exercise.sets = sets_min
            changes.append(f"sets {s}->{sets_min}")
        elif s > sets_max:
            exercise.sets = sets_max
            changes.append(f"sets {s}->{sets_max}")

    reps_min, reps_max = _envelope_pair(dr, "reps_min", "reps_max")
    if reps_min is not None and reps_max is not None and exercise.reps:
        r = int(exercise.reps)
        if r < reps_min:
            exercise.reps = reps_min
            changes.append(f"reps {r}->{reps_min}")
        elif r > reps_max:
            exercise.reps = reps_max
            changes.append(f"reps {r}->{reps_max}")

    if changes:
        result.clamps.append(
            DoseClamp(
                exercise_id=str(exercise.exercise_id),
                changes=changes,
            )
        )


def _handle_duration_based(
    exercise: Exercise,
    dr: dict[str, Any],
    result: SafetyCheckResult,
) -> None:
    """Duration-based exercise: clamp duration_minutes to [duration_min, duration_max].

    If duration_minutes missing entirely, synthesize a default (warn-only — the
    LLM gave wrong dosing schema, not a patient-safety failure). Strip stray
    sets/reps so downstream consumers don't render confusing dose strings.
    """
    dur_min, dur_max = _envelope_pair(dr, "duration_min", "duration_max")

    if exercise.duration_minutes is None:
        default_duration = dur_min if dur_min else 10
        exercise.duration_minutes = default_duration
        if exercise.reps and exercise.reps != 1:
            exercise.reps = None
        if exercise.sets and exercise.sets > 1:
            exercise.sets = 1
        result.synthesis.append(
            DoseSynthesis(
                exercise_id=str(exercise.exercise_id),
                reason="dose_schema_mismatch",
                detail=f"duration-based exercise; synthesized duration_minutes={default_duration}",
            )
        )
        return

    d = int(exercise.duration_minutes)
    if dur_min is not None and d < dur_min:
        exercise.duration_minutes = dur_min
        result.clamps.append(
            DoseClamp(
                exercise_id=str(exercise.exercise_id),
                changes=[f"duration {d}->{dur_min}"],
            )
        )
    elif dur_max is not None and d > dur_max:
        exercise.duration_minutes = dur_max
        result.clamps.append(
            DoseClamp(
                exercise_id=str(exercise.exercise_id),
                changes=[f"duration {d}->{dur_max}"],
            )
        )


def _handle_education(
    exercise: Exercise,
    dr: dict[str, Any],
    result: SafetyCheckResult,
) -> None:
    """Single-instance instructional exercise (reps_max=1 AND no duration).

    Normalize to sets=1, reps=1. Not a safety violation — the LLM picked the
    wrong dosing schema for an instructional item.
    """
    sets_max = dr.get("sets_max", 1)
    try:
        sets_max_int = int(sets_max)
    except (ValueError, TypeError):
        sets_max_int = 1

    changed = False
    if exercise.sets and exercise.sets > sets_max_int:
        exercise.sets = sets_max_int
        changed = True
    if exercise.reps and exercise.reps != 1:
        exercise.reps = 1
        changed = True

    if changed:
        result.synthesis.append(
            DoseSynthesis(
                exercise_id=str(exercise.exercise_id),
                reason="non_rep_based_normalized",
                detail="education/technique exercise; normalized to single-instance schema",
            )
        )


# ──────────────────────────────────────────────────────────────────────────
# Stage: invented (no-registry) exercise — generic ceilings + name/notes scan
# ──────────────────────────────────────────────────────────────────────────


def _invented_id(exercise: Exercise) -> str:
    """Stable, non-PHI identifier for an invented exercise in safety metadata.

    Invented exercises have `exercise_id=None` by the time the safety check runs
    (resolve already nulled it), so we key clamps/hits on the exercise name —
    the clinical label is not PHI.
    """
    return exercise.name or "invented"


def _clamp_invented(exercise: Exercise, result: SafetyCheckResult) -> None:
    """Clamp an AI-invented exercise to generic absolute dose ceilings.

    Each field is guarded with `is not None` before `int(...)` — a duration-only
    invented exercise has sets/reps == None. Records one DoseClamp listing every
    field clamped. Within-bounds values are untouched.
    """
    changes: list[str] = []

    if exercise.sets is not None:
        s = int(exercise.sets)
        clamped = _clamp_value(s, _INVENTED_SETS_MIN, _INVENTED_SETS_MAX)
        if clamped != s:
            exercise.sets = clamped
            changes.append(f"sets {s}->{clamped}")

    if exercise.reps is not None:
        r = int(exercise.reps)
        clamped = _clamp_value(r, _INVENTED_REPS_MIN, _INVENTED_REPS_MAX)
        if clamped != r:
            exercise.reps = clamped
            changes.append(f"reps {r}->{clamped}")

    if exercise.duration_minutes is not None:
        d = int(exercise.duration_minutes)
        clamped = _clamp_value(d, _INVENTED_DURATION_MIN, _INVENTED_DURATION_MAX)
        if clamped != d:
            exercise.duration_minutes = clamped
            changes.append(f"duration {d}->{clamped}")

    if changes:
        result.clamps.append(DoseClamp(exercise_id=_invented_id(exercise), changes=changes))


def _scan_invented_contraindications(
    exercise: Exercise,
    diag_terms: list[str],
    result: SafetyCheckResult,
) -> None:
    """Token-overlap the patient's diagnosis terms against the invented
    exercise's own name + notes (there is no registry contraindication list).

    Mirrors `_scan_contraindications` matching, but the "contraindication" text
    is the exercise label/notes itself — a diagnosis-term match means the
    invented exercise references the patient's condition and the doctor should
    confirm it is appropriate. Records one ContraindicationHit on overlap.
    """
    if not diag_terms:
        return

    text = f"{exercise.name} {exercise.notes}".strip()
    if not text:
        return

    diag_tokens = {tok for term in diag_terms for tok in _tokenize(term)}
    if not diag_tokens:
        return

    ex_tokens = set(_tokenize(text))
    overlap = diag_tokens & ex_tokens
    if not overlap:
        return

    matched_term = next(
        (t for t in diag_terms if any(tok in _tokenize(t) for tok in overlap)),
        next(iter(overlap)),
    )
    result.contraindications.append(
        ContraindicationHit(
            exercise_id=_invented_id(exercise),
            diagnosis_term=matched_term,
            contraindication=text,
        )
    )


def _clamp_value(value: int, lo: int, hi: int) -> int:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# ──────────────────────────────────────────────────────────────────────────
# Stage: contraindications
# ──────────────────────────────────────────────────────────────────────────


def _scan_contraindications(
    exercise: Exercise,
    registry: RegistryExercise,
    diag_terms: list[str],
    result: SafetyCheckResult,
) -> None:
    """Word-level token overlap between patient diagnosis text and the registry
    contraindication strings. Substring match would miss `post-op knee fracture`
    vs `unstable knee fracture`; ICD codes are matched as whole tokens.
    """
    if not diag_terms or not registry.contraindications:
        return

    # Harm-asymmetry (same rationale as `_allergen_match` below): a MISSED
    # contraindication is real patient harm; an extra flag is cheap (the scan
    # never strips the plan, only surfaces it for the doctor). The ≥4-char
    # `_tokenize` path silently drops safety-critical SHORT clinical terms
    # ("hip", "ACL", "DVT", "ORIF", "TKR", …), so a "hip precautions"
    # contraindication could never match a "hip replacement" diagnosis. Add a
    # curated short-term allowlist on BOTH sides so those terms survive.
    def _terms(text: str) -> set[str]:
        return set(_tokenize(text)) | _critical_short_terms(text)

    diag_tokens = {tok for term in diag_terms for tok in _terms(term)}
    if not diag_tokens:
        return

    for contra in registry.contraindications:
        contra_str = str(contra).lower()
        if not contra_str:
            continue
        contra_tokens = _terms(contra_str)
        overlap = diag_tokens & contra_tokens
        if overlap:
            matched_term = next(
                (t for t in diag_terms if any(tok in _terms(t) for tok in overlap)),
                next(iter(overlap)),
            )
            result.contraindications.append(
                ContraindicationHit(
                    exercise_id=str(exercise.exercise_id),
                    diagnosis_term=matched_term,
                    contraindication=str(contra),
                )
            )
            return


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def _envelope_pair(dr: dict[str, Any], min_key: str, max_key: str) -> tuple[int | None, int | None]:
    lo = _safe_int(dr.get(min_key))
    hi = _safe_int(dr.get(max_key))
    return lo, hi


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


_TOKEN_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "post",
        "post-op",
        "without",
    }
)


def _tokenize(text: str) -> list[str]:
    """Split free-text into clinical tokens — alpha words ≥4 chars or ICD codes.

    Skips short conjunctions and "post-op"-style prefixes that would
    otherwise produce false positives.
    """
    out: list[str] = []
    for raw in text.lower().replace("/", " ").replace("-", " ").split():
        token = raw.strip(".,;:()[]'\"")
        if not token:
            continue
        if token in _TOKEN_STOPWORDS:
            continue
        if len(token) >= 4:
            out.append(token)
    return out


# Safety-critical clinical terms shorter than the ≥4-char `_tokenize` floor.
# Curated (not "all short words") so the contraindication scan stays precise —
# these are anatomical/diagnostic abbreviations whose omission would silently
# defeat a real contraindication match. Extend deliberately, with a test.
_CRITICAL_SHORT_TERMS = frozenset(
    {
        # joints / anatomy
        "hip",
        "acl",
        "pcl",
        "mcl",
        "lcl",
        "sci",
        "tbi",
        # arthroplasty / fixation
        "tka",
        "tkr",
        "tha",
        "thr",
        "orif",
        # vascular / cardiac
        "dvt",
        "pe",
        "vte",
        "mi",
        "acs",
        "cad",
        "chf",
        "htn",
        "af",
        "tia",
        # organ-system disease
        "ckd",
        "esrd",
        "copd",
        "ild",
        # rheum / neuro
        "ra",
        "oa",
        "als",
    }
)


def _critical_short_terms(text: str) -> set[str]:
    """Curated short clinical terms (2–3 chars) present in ``text``.

    Complements ``_tokenize`` (which drops <4-char tokens) so the
    contraindication scan does not silently miss "hip", "ACL", "DVT", etc.
    """
    found: set[str] = set()
    for raw in text.lower().replace("/", " ").replace("-", " ").split():
        token = raw.strip(".,;:()[]'\"")
        if token in _CRITICAL_SHORT_TERMS:
            found.add(token)
    return found


def _diagnosis_terms(patient: PatientSnapshot | None) -> list[str]:
    if patient is None:
        return []
    raw: list[str] = []
    if patient.diagnosis:
        raw.append(patient.diagnosis)
    raw.extend(patient.comorbidities or [])
    return [t.lower().strip() for t in raw if t and t.strip()]


# ──────────────────────────────────────────────────────────────────────────
# Stage: allergy cross-check
# ──────────────────────────────────────────────────────────────────────────


def _allergen_match(allergen: str, haystack: str) -> bool:
    """Plain substring containment of `allergen` inside `haystack` (both lowered).

    Deliberately NOT the `_tokenize` ≥4-char path nor a word-boundary regex —
    short allergens ("egg", "soy", "nut") and their plurals/compounds ("eggs",
    "walnut", "soybean oil") must all match. A false-positive flag is cheap
    (the scan never strips the plan, only flags it for the doctor), while a
    missed allergen is the real harm — so the harm asymmetry favors substring.
    """
    if not allergen or not haystack:
        return False
    return allergen in haystack


def _record_allergen_hits(
    allergens: list[str],
    haystack: str,
    where: str,
    label: str,
    result: SafetyCheckResult,
    seen: set[tuple[str, str, str]],
) -> None:
    if not haystack.strip():
        return
    lowered = haystack.lower()
    for allergen in allergens:
        if not _allergen_match(allergen, lowered):
            continue
        key = (allergen, where, label)
        if key in seen:
            continue
        seen.add(key)
        result.allergy_hits.append(AllergyHit(allergen=allergen, where=where, matched=label))


# ── Free-text allergy extraction (universal, substance-agnostic) ────────────
# Extracts allergens the clinician stated in the free-text case ("Аллергия на
# диклофенак", "allergic to penicillin and sulfa") so a documented allergy is
# honoured even when it was NOT entered in the structured allergy field. This is
# NOT hardcoded clinical knowledge: it keys off explicit ALLERGY MARKERS and
# captures whatever substance the clinician named — ANY drug/food/agent, never a
# curated allow-list. The reaction description is never captured as the allergen.
_ALLERGY_MARKERS = re.compile(
    r"аллерги\w*\s+на\s+"  # ru: "аллергия на X"
    r"|аллерги\w*\s*[:\-—]\s*"  # ru: "аллергия: X"
    r"|allerg(?:y|ic)\s+to\s+"  # en: "allergic to X" / "allergy to X"
    r"|allerg(?:y|ies)\s*[:\-—]\s*"  # en: "allergy: X"
    r"|allergiya\w*\s*[:\-—]\s*"  # uz: "allergiya: X"
    r"|allergiya\w*\s+bor\s+",  # uz: "... allergiya bor" — handled loosely
    re.IGNORECASE | re.UNICODE,
)
# The reaction is delimited by a dash / paren / sentence end — never by a
# connector. Connectors (and/и/va/comma/semicolon) join MULTIPLE allergens, so
# they must NOT terminate the capture (else a second allergen is silently lost).
_ALLERGEN_BOUNDARY = re.compile(r"[—()\.;\n]|\s-\s", re.UNICODE)
_ALLERGEN_CONNECTOR = re.compile(r"\s*(?:[,;]|\bи\b|\band\b|\bva\b)\s*", re.IGNORECASE | re.UNICODE)
# A negation immediately before the marker DENIES an allergy ("no allergy to…",
# "без аллергии", "аллергий нет") — such a match must not extract an allergen.
_ALLERGEN_NEGATION = re.compile(
    r"(?:\bno\b|\bnot\b|\bnkda\b|\bdenies\b|\bбез\b|\bне\b|\bнет\b|\byo['’]?q\b)\W*$",
    re.IGNORECASE | re.UNICODE,
)
_ALLERGEN_MAX_LEN = 40


def extract_case_allergens(case_text: str | None) -> list[str]:
    """Allergens explicitly stated in the free-text case. Universal / marker-based.

    Substance-agnostic: captures whatever the clinician named after an allergy
    marker (any drug, food, or agent), splits multi-allergen lists on connectors,
    skips negated statements, and never captures the reaction description. Returns
    a lowercased, de-duplicated list; ``[]`` when nothing is stated.
    """
    if not case_text:
        return []
    out: list[str] = []
    for mo in _ALLERGY_MARKERS.finditer(case_text):
        pre = case_text[max(0, mo.start() - 14) : mo.start()]
        if _ALLERGEN_NEGATION.search(pre):
            continue
        rest = case_text[mo.end() :]
        boundary = _ALLERGEN_BOUNDARY.search(rest)
        phrase = (rest[: boundary.start()] if boundary else rest).strip()
        if not phrase or len(phrase) > _ALLERGEN_MAX_LEN:
            continue
        for part in _ALLERGEN_CONNECTOR.split(phrase):
            token = part.strip().lower()
            if token and len(token) <= _ALLERGEN_MAX_LEN:
                out.append(token)
    seen: set[str] = set()
    return [a for a in out if not (a in seen or seen.add(a))]


def merge_allergens(patient: PatientSnapshot | None, case_text: str | None) -> list[str]:
    """Unified allergen list: structured record ∪ free-text case, de-duplicated.

    The single source the cross-check and the surfacing both consume, so a
    documented allergy is honoured no matter WHERE it was recorded.
    """
    merged: list[str] = []
    if patient is not None and patient.allergies:
        merged.extend(a for a in patient.allergies if a and a.strip())
    merged.extend(extract_case_allergens(case_text))
    seen: set[str] = set()
    return [a for a in merged if not (a.lower() in seen or seen.add(a.lower()))]


def _scan_allergies(
    plan: GeneratedPlan,
    allergens: list[str],
    result: SafetyCheckResult,
) -> None:
    """Token-match each patient allergen against everything the plan prescribes:
    exercise name/notes, recommended supplements, diet-plan foods, and the
    medication schedule. Flag-only — the doctor confirms; nothing is stripped.

    ``allergens`` is the UNIFIED allergen list (structured record + any explicitly
    stated in the free-text case), substance-agnostic — never a curated subset.
    """
    allergens = [a.lower().strip() for a in (allergens or []) if a and a.strip()]
    if not allergens:
        return

    seen: set[tuple[str, str, str]] = set()

    for phase in plan.phases:
        for ex in phase.exercises:
            _record_allergen_hits(
                allergens,
                f"{ex.name} {ex.notes}",
                "exercise",
                ex.name or "exercise",
                result,
                seen,
            )

    for supp in plan.diet_plan.supplements:
        _record_allergen_hits(allergens, supp.name, "supplement", supp.name, result, seen)

    for category in plan.diet_plan.recommended_foods:
        food_text = f"{category.category} {category.examples}"
        _record_allergen_hits(allergens, food_text, "food", category.category, result, seen)

    for med in plan.medication_schedule:
        _record_allergen_hits(allergens, med.medication, "medication", med.medication, result, seen)


# ──────────────────────────────────────────────────────────────────────────
# Stage: surface documented allergies to the patient/doctor
# ──────────────────────────────────────────────────────────────────────────

# Localized label + instruction for the deterministic documented-allergy alert.
# The allergen NAME is inserted verbatim (a substance/drug name is not
# translated); only the surrounding wording is localized to the plan's language.
# Safety-critical, so every supported plan language is covered (no fallback to a
# single default the way generic caution scaffolding does).
_ALLERGY_ALERT_LABELS: dict[str, tuple[str, str]] = {
    "en": (
        "Documented allergy",
        "Recorded patient allergy: {name}. Do not prescribe or administer this or "
        "related agents; verify before any new medication.",
    ),
    "ru": (
        "Задокументированная аллергия",
        "Зарегистрированная аллергия пациента: {name}. Не назначать этот и "
        "родственные препараты; проверять перед любым новым лекарством.",
    ),
    "uz": (
        "Hujjatlashtirilgan allergiya",
        "Bemorning qayd etilgan allergiyasi: {name}. Ushbu va shunga o'xshash "
        "vositalarni buyurmang; har qanday yangi doridan oldin tekshiring.",
    ),
    "uz-cyrl": (
        "Ҳужжатлаштирилган аллергия",
        "Беморнинг қайд этилган аллергияси: {name}. Ушбу ва шунга ўхшаш "
        "воситаларни буюрманг; ҳар қандай янги доридан олдин текширинг.",
    ),
}


def surface_documented_allergies(plan: GeneratedPlan, allergens: list[str]) -> int:
    """Deterministically surface a patient's documented allergies in the plan.

    ``allergens`` is the UNIFIED allergen list (structured record ∪ free-text
    case — build it with ``merge_allergens``). Sourced from the record/case, NOT
    the LLM, so a documented allergy can never be silently dropped from the plan
    the doctor and patient see. Appends one localized ``SafetyAlert`` per allergen
    to ``plan.safety_alerts`` (which both the doctor-review composer and the
    patient plan view already render). Returns the number of alerts added.

    Idempotent / non-duplicating: an allergen already named in an existing alert
    (e.g. one the LLM surfaced, or a prior pass) is skipped, so regeneration and
    the LLM's own output never double-list the same allergy. No-op when the list
    is empty. The allergen name is patient-facing here (the whole point is that
    the patient sees their allergy) but is still never written to logs.
    """
    allergens = [a.strip() for a in (allergens or []) if a and a.strip()]
    if not allergens:
        return 0

    lang = getattr(plan, "output_language", "") or ""
    lang = getattr(lang, "value", lang)  # tolerate a PlanLanguage enum or a bare str
    label, detail_tmpl = _ALLERGY_ALERT_LABELS.get(str(lang), _ALLERGY_ALERT_LABELS["en"])

    # Lowered text of every existing alert, so an allergen the LLM already named
    # (or a prior surfacing pass) is not surfaced twice.
    existing_text = " ".join(f"{sa.alert} {sa.detail}" for sa in plan.safety_alerts).lower()

    added = 0
    for allergen in allergens:
        if allergen.lower() in existing_text:
            continue
        plan.safety_alerts.append(
            SafetyAlert(alert=label, detail=detail_tmpl.format(name=allergen))
        )
        existing_text += f" {label} {allergen}".lower()
        added += 1
    return added


# ──────────────────────────────────────────────────────────────────────────
# Stage: medication dose reconciliation (deterministic, universal)
# ──────────────────────────────────────────────────────────────────────────

# A dose stated as a NUMERIC RANGE with a strength unit ("2.5-5 mg", "2.5–5 мг",
# "5 to 10 mcg") is never a safe FINAL medication instruction: the patient takes
# ONE specific dose, and choosing between the ends is a prescriber decision the
# plan must not present as settled. This detection is purely FORMAT-based and
# therefore universal — it encodes NO drug-specific dosing knowledge (it does not
# know apixaban's reduction criteria), only that an ambiguous dose must be
# reconciled, so it holds for every medication in every field. Strength units
# only (not "tablet"/"таб") so a legitimate COUNT range like "1-2 tablets PRN"
# is not falsely flagged.
#
# A RANGE is separated by a dash / "to" / "до" — NEVER by "/". A slash between two
# strengths is the fixed-dose-COMBINATION ratio of a single product (levodopa/
# carbidopa "100/25 mg", co-careldopa "25/100 mg", co-amoxiclav "500/125 mg"): the
# patient takes that exact combined tablet, so it must NOT be flagged as an ambiguous
# dose to reconcile. Excluding "/" removes that false-positive class without missing
# a real range (ranges never use "/").
_DOSE_STRENGTH_UNIT = r"(?:mg|mcg|µg|ug|g|ml|units?|iu|мг|мкг|г|мл|ед)"
# A range dose is "<n> [unit] <sep> <n> unit" where <sep> is a dash / "to" / "до"
# (NEVER "/", which is a fixed-dose combination ratio — see below). The FIRST unit is
# OPTIONAL so BOTH the compact form ("2.5-5 mg") and the equally-common EXPANDED form
# where the unit is repeated after each number ("2.5 mg – 5 mg", "2.5 mg to 5 mg") are
# caught; a trailing unit after the second number is always required so a bare count
# range ("1-2 tablets") is not mis-flagged as a strength range.
_AMBIGUOUS_DOSE_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:" + _DOSE_STRENGTH_UNIT + r"\b\s*)?"
    r"(?:[-–—]|\bto\b|\bдо\b)\s*\d+(?:[.,]\d+)?\s*" + _DOSE_STRENGTH_UNIT + r"\b",
    re.IGNORECASE,
)

_MED_DOSE_RECON_LABELS: dict[str, tuple[str, str]] = {
    "en": (
        "Confirm medication dose",
        "{name}: the plan lists an ambiguous dose ({dose}). Continue the currently "
        "prescribed dose and reconcile the exact dose with the treating physician "
        "before delivery — do not self-select between the values.",
    ),
    "ru": (
        "Уточните дозу препарата",
        "{name}: в плане указана неоднозначная доза ({dose}). Продолжайте текущую "
        "назначенную дозу и согласуйте точную дозу с лечащим врачом до выдачи — не "
        "выбирайте самостоятельно между значениями.",
    ),
    "uz": (
        "Dori dozasini tasdiqlang",
        "{name}: rejada noaniq doza ko'rsatilgan ({dose}). Hozirgi buyurilgan dozani "
        "davom ettiring va yetkazishdan oldin aniq dozani davolovchi shifokor bilan "
        "kelishing — qiymatlar orasidan o'zingiz tanlamang.",
    ),
    "uz-cyrl": (
        "Дори дозасини тасдиқланг",
        "{name}: режада ноаниқ доза кўрсатилган ({dose}). Ҳозирги буюрилган дозани "
        "давом эттиринг ва етказишдан олдин аниқ дозани даволовчи шифокор билан "
        "келишинг — қийматлар орасидан ўзингиз танламанг.",
    ),
}


def flag_ambiguous_medication_doses(plan: GeneratedPlan) -> list[str]:
    """Deterministically flag any medication stated with an ambiguous RANGE dose.

    Roots the recurring "apixaban 2.5–5 mg" failure at the GATE, not the prompt:
    a range dose reaches the doctor as a definite, mandatory reconciliation
    SafetyAlert that the LLM cannot omit or soften, and a 'proceed' plan is
    escalated to 'caution' so it is triaged for review. Warn, never block — the
    medication is never removed. Purely format-based + universal (no drug-specific
    knowledge). Idempotent: a med already carrying this reconciliation alert is
    not flagged twice. Returns the flagged medication labels (audit/meta). No-op
    when every dose is unambiguous or there is no medication schedule.
    """
    meds = getattr(plan, "medication_schedule", None) or []
    if not meds:
        return []

    lang = getattr(plan, "output_language", "") or ""
    lang = getattr(lang, "value", lang)
    label, detail_tmpl = _MED_DOSE_RECON_LABELS.get(str(lang), _MED_DOSE_RECON_LABELS["en"])

    existing_text = " ".join(f"{sa.alert} {sa.detail}" for sa in plan.safety_alerts).lower()
    flagged: list[str] = []
    for entry in meds:
        name = (getattr(entry, "medication", "") or "").strip()
        dose = (getattr(entry, "dose", "") or "").strip()
        haystack = f"{name} {dose}".strip()
        match = _AMBIGUOUS_DOSE_RE.search(haystack)
        if not match:
            continue
        rng = re.sub(r"\s+", " ", match.group(0)).strip()
        med_label = name or haystack
        # Idempotent: skip if this med already carries OUR reconciliation alert.
        if label.lower() in existing_text and med_label.lower() in existing_text:
            continue
        plan.safety_alerts.append(
            SafetyAlert(alert=label, detail=detail_tmpl.format(name=med_label, dose=rng))
        )
        existing_text += f" {label} {med_label} {rng}".lower()
        flagged.append(med_label)

    # Escalate a proceeding plan so the ambiguous dose is triaged for doctor
    # review. Never downgrade an abstaining verdict or an existing caution.
    if flagged and not plan.verdict.abstains and plan.verdict != Verdict.CAUTION:
        plan.verdict = Verdict.CAUTION
        plan.decision = "caution"

    return flagged


__all__ = [
    "AllergyHit",
    "ContraindicationHit",
    "DoseClamp",
    "DoseSynthesis",
    "SafetyCheckResult",
    "flag_ambiguous_medication_doses",
    "run_safety_check",
    "surface_documented_allergies",
]
