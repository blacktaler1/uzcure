"""Patient-data fidelity gate — universal source-of-truth check.

UzCure/UzCure is universal across medical fields, so this gate is deliberately
condition-agnostic: it never encodes a per-specialty rule. It enforces ONE
invariant that holds for every field — the plan must not assert a patient-
specific fact that contradicts the source case.

The first (and highest-severity) fidelity check is the OPERATED / AFFECTED SIDE.
An LLM that flips the laterality — e.g. the case documents a LEFT hip
arthroplasty but the plan prescribes RIGHT-side exercises — is a direct
patient-specific failure (external clinical audit, 2026-07). This is caught
deterministically here, independent of the model, and surfaced as an escalation
reason so a doctor must reconcile it before the plan reaches the patient. The
gate NEVER blocks or silently rewrites the plan (a side it flags could be a
legitimate contralateral exercise); it flags for human review — the
SaMD-appropriate, human-in-the-loop behavior.

Design notes (why this is safe to run on every generation):
  * Multilingual: recognises left/right in English, Russian and Uzbek — the
    three platform languages — using anatomical-adjective forms so everyday
    words ("right" as in correct, "право" as in rights) do not trip it.
  * Fail-open on ambiguity: it flags ONLY when the case commits to exactly ONE
    side and the plan commits to the OPPOSITE side. A case that names both sides,
    or none, produces no flag — a false negative (missing a real mismatch) is the
    dangerous direction, but ambiguity is resolved toward not crying wolf while
    still catching the unambiguous contradiction the audit found.
  * Bilateral / unspecified exercises never trip it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing only
    from app.features.plan_generation.domain.entities import GeneratedPlan

# ── Multilingual laterality lexicon ─────────────────────────────────────────
# Anatomical-adjective forms only, so "right" (correct) / "право" (a legal
# right) / "o'ng" ambiguities are minimised. Word-boundary anchored.
_LEFT_PATTERNS = (
    r"\bleft\b",  # en
    r"\bлев(?:ый|ого|ом|ому|ым|ой|ая|ую|ое|ые|ых|ыми)\b",  # ru adjective forms
    r"\bслева\b",  # ru "on the left"
    r"\bchap\b",  # uz
)
_RIGHT_PATTERNS = (
    r"\bright\b",  # en
    r"\bправ(?:ый|ого|ом|ому|ым|ой|ая|ую|ое|ые|ых|ыми)\b",  # ru adjective forms
    r"\bсправа\b",  # ru "on the right"
    r"\bo['ʻ‘’]?ng\b",  # uz o'ng / oʻng / ong
)

_LEFT_RE = re.compile("|".join(_LEFT_PATTERNS), re.IGNORECASE | re.UNICODE)
_RIGHT_RE = re.compile("|".join(_RIGHT_PATTERNS), re.IGNORECASE | re.UNICODE)


def _sides_in_text(text: str) -> set[str]:
    """The anatomical side(s) the text commits to: subset of {'left','right'}."""
    if not text:
        return set()
    sides: set[str] = set()
    if _LEFT_RE.search(text):
        sides.add("left")
    if _RIGHT_RE.search(text):
        sides.add("right")
    return sides


def _plan_sides(plan: "GeneratedPlan") -> set[str]:
    """Distinct side-specific laterality values the plan's exercises commit to.

    Only 'left'/'right' count — 'bilateral', 'both', and '' (n/a) are never a
    mismatch. The ``laterality`` field is a free-form string and, under the hard
    output-language contract, arrives localised ("левая", "chap", "o'ng") — so it
    is normalised through the SAME multilingual lexicon used for the case, not
    matched against the two English literals (which the model rarely emits when
    told to write every field in Russian/Uzbek). A value naming exactly one side
    contributes that side; a value naming both (or neither) contributes nothing.
    """
    sides: set[str] = set()
    for phase in getattr(plan, "phases", []) or []:
        for ex in getattr(phase, "exercises", []) or []:
            lat = (getattr(ex, "laterality", "") or "").strip()
            if not lat:
                continue
            found = _sides_in_text(lat)
            # A laterality string that commits to exactly one side is a real,
            # side-specific prescription; 'bilateral'/'both'/ambiguous → skip.
            if len(found) == 1:
                sides |= found
    return sides


def build_fidelity_case_text(
    case_text: str | None,
    *,
    diagnosis: str | None = None,
    comorbidities: list[str] | None = None,
) -> str:
    """The source-of-truth corpus the fidelity gates check the plan against.

    The operated/affected SIDE (and documented labs/comorbidities) frequently
    live in the STRUCTURED diagnosis the clinician entered ("Left TKA", "правый
    ТБС") while the free-text case omits them. Checking only ``case_history_text``
    then misses a contralateral plan entirely — reopening the audited bug through
    the diagnosis field. So the gates check the plan against the same documented
    context the model was given: case text + diagnosis + comorbidities.
    """
    parts: list[str] = []
    if case_text:
        parts.append(str(case_text))
    if diagnosis:
        parts.append(str(diagnosis))
    if comorbidities:
        parts.extend(str(c) for c in comorbidities if c)
    return "\n".join(parts)


def check_operated_side_fidelity(plan: "GeneratedPlan", case_text: str) -> dict | None:
    """Deterministic operated/affected-side fidelity check.

    Returns a PHI-free violation dict when the case commits to exactly ONE side
    and the plan prescribes side-specific exercises on the OPPOSITE side; ``None``
    otherwise. The dict is language-agnostic and safe to store/surface:

        {"case_side": "left", "plan_sides": ["right"], "conflicting": ["right"]}
    """
    case_sides = _sides_in_text(case_text or "")
    if len(case_sides) != 1:
        # No side, or both sides, documented → ambiguous, do not flag.
        return None
    case_side = next(iter(case_sides))
    opposite = "right" if case_side == "left" else "left"

    plan_sides = _plan_sides(plan)
    if opposite in plan_sides:
        return {
            "case_side": case_side,
            "plan_sides": sorted(plan_sides),
            "conflicting": [opposite],
        }
    return None


def _plan_diagnosis_text(plan: "GeneratedPlan") -> str:
    """The plan's OWN stated diagnosis line (``plan_metadata.diagnosis_short``).

    This is the plan's headline condition — e.g. "Состояние после ТЭП левого ТБС".
    It is the authoritative place the plan commits to the operated side, distinct
    from per-exercise laterality (which may legitimately be contralateral).
    """
    meta = getattr(plan, "plan_metadata", None)
    if meta is None:
        return ""
    return (getattr(meta, "diagnosis_short", "") or "").strip()


def check_diagnosis_laterality_conflict(plan: "GeneratedPlan", case_text: str) -> dict | None:
    """FAIL-CLOSED laterality guard: plan's stated DIAGNOSIS vs the case side.

    Compares the operated/affected side named in the case against the side named
    in the plan's own diagnosis line (``plan_metadata.diagnosis_short``) — NOT the
    per-exercise laterality. A plan whose *diagnosis* names the OPPOSITE side from
    the case is a never-event (a contralateral procedure statement) and must BLOCK
    generation, not merely escalate for review.

    Deliberately narrower than ``check_operated_side_fidelity`` (the flag-only,
    per-exercise signal): a single-side case may legitimately include a
    contralateral supporting-limb exercise, so exercise laterality must never
    hard-block. This fires ONLY when BOTH the case AND the diagnosis commit to
    exactly one side AND those sides are opposite — so it cannot false-block an
    ambiguous, side-less, or matching plan. Deterministic and model-independent.

    Returns a PHI-free dict on conflict (safe to log/store), else ``None``:

        {"case_side": "left", "diagnosis_side": "right"}
    """
    case_sides = _sides_in_text(case_text or "")
    if len(case_sides) != 1:
        return None
    diag_sides = _sides_in_text(_plan_diagnosis_text(plan))
    if len(diag_sides) != 1:
        return None
    case_side = next(iter(case_sides))
    diag_side = next(iter(diag_sides))
    if case_side != diag_side:
        return {"case_side": case_side, "diagnosis_side": diag_side}
    return None


# ── Fabricated-lab gate ─────────────────────────────────────────────────────
# The audit found the plan asserting lab VALUES the case never provided (Hb 104,
# CRP 18, eGFR 58, HbA1c 8.1, D-dimer 980, vitamin D 18, ...) and reasoning from
# them as fact. This gate is universal: a curated multilingual lexicon of common
# lab NAMES (not a per-specialty rule). It flags a lab ONLY when the plan asserts
# a NUMERIC value for it AND the case never mentions that lab — i.e. an invented
# measurement. A lab named in the case (any value) is trusted; a generic mention
# with no number ("monitor renal function") is not a fabricated value.
# EN + RU + Uzbek-Latin. The Uzbek forms are the standard Latin transliterations of
# the SAME Russian terms already trusted here (гемоглобин→gemoglobin, креатинин→
# kreatinin, калий→kaliy, натрий→natriy, лейкоцит→leykotsit, тромбоцит→trombotsit),
# not new clinical concepts — Uzbek is a first-class output language, and without
# them an Uzbek plan could invent a lab value that slipped the gate. ferritin /
# troponin / and the international abbreviations (hb, crp, inr, egfr, wbc, bnp,
# d-dimer) are already language-agnostic here.
_LAB_LEXICON: dict[str, str] = {
    "hemoglobin": r"h[aeo]*moglobin|gemoglobin|\bhb\b|\bhgb\b|гемоглобин",
    "crp": r"\bcrp\b|c-?reactive protein|срб|с-?реактивн",
    "esr": r"\besr\b|\bсоэ\b",
    "egfr": r"\begfr\b|\bскф\b|glomerular filtration",
    "hba1c": r"hba1c|\ba1c\b|гликированн",
    "d_dimer": r"d-?dimer|д-?димер",
    "creatinine": r"creatinin|kreatinin|креатинин",
    "inr": r"\binr\b|\bмно\b",
    "vitamin_d": r"vitamin\s?d\b|витамин\s?d\b|25-oh",
    "ferritin": r"ferritin|ферритин",
    "platelets": r"platelet|trombotsit|тромбоцит",
    "wbc": r"\bwbc\b|leukocyt|leykotsit|лейкоцит",
    "potassium": r"potassium|kaliy|калий",
    "sodium": r"sodium|natriy|натрий",
    "troponin": r"troponin|тропонин",
    "bnp": r"\bnt-?probnp\b|\bbnp\b",
}
_LAB_COMPILED = {k: re.compile(v, re.IGNORECASE | re.UNICODE) for k, v in _LAB_LEXICON.items()}
_DIGIT_RE = re.compile(r"\d")


def _plan_text(plan: "GeneratedPlan") -> str:
    """All text the plan emitted, for scanning. Comprehensive (every field)."""
    try:
        return plan.model_dump_json()
    except Exception:  # pragma: no cover — defensive; tests use real/JSON stubs
        try:
            import json

            return json.dumps(plan.to_api_dict(), ensure_ascii=False)
        except Exception:
            return str(plan)


def check_fabricated_labs(plan: "GeneratedPlan", case_text: str) -> list[str]:
    """Canonical lab names the plan asserts a numeric value for that are NOT in
    the case (i.e. invented measurements). Empty list when the plan invents none.
    """
    text = _plan_text(plan)
    case = case_text or ""
    invented: list[str] = []
    for canonical, pat in _LAB_COMPILED.items():
        for m in pat.finditer(text):
            # A value is "asserted" when a digit sits just after (e.g. "Hb 104")
            # or just before (e.g. "104 g/l Hb") the lab token.
            after = text[m.end() : m.end() + 12]
            before = text[max(0, m.start() - 8) : m.start()]
            if _DIGIT_RE.search(after) or _DIGIT_RE.search(before):
                if not pat.search(case):
                    invented.append(canonical)
                break  # one asserted value per lab is enough to flag it
    return sorted(set(invented))


# ── Omitted-medication gate ─────────────────────────────────────────────────
# A current medication the clinician recorded that silently vanishes from the
# plan's medication_schedule disappears from the patient's at-home reminders — a
# universal, deterministic-detectable, high-harm failure (the prompt makes
# enumeration a "HARD REQUIREMENT" but that is model-trusted). This gate compares
# the STRUCTURED, clinician-entered known-medication list against the scheduled
# drugs and flags any that are missing. Flag-only (escalate for doctor review);
# it NEVER auto-inserts a drug (that would be clinical authorship).
_DRUG_HEAD_SPLIT = re.compile(r"\d")


def _drug_head(med_text: str) -> str:
    """The drug NAME from a free-form medication string, minus dose/frequency.

    "Rivaroxaban 10 mg daily" → "rivaroxaban"; "insulin glargine 20 units" →
    "insulin glargine"; "Метформин 500 мг" → "метформин". The name is the text
    before the first digit (the dose), which handles multi-word drug names.
    """
    s = str(med_text or "").strip()
    if not s:
        return ""
    m = _DRUG_HEAD_SPLIT.search(s)
    head = s[: m.start()] if m else s
    return head.strip(" ,.;:-–—()").strip().lower()


def check_medication_omissions(
    plan: "GeneratedPlan", known_medications: list[str] | None
) -> list[str]:
    """Drug names from ``known_medications`` that do NOT appear in the plan's
    ``medication_schedule`` (silently dropped). Empty list when none are omitted.

    Conservative substring match in BOTH directions (generic/brand and dose
    noise), and drug heads shorter than 3 chars are skipped to avoid trivial
    false matches. A false positive only prompts the doctor to double-check (safe
    direction); a false negative — the dangerous direction — is what this guards.
    """
    if not known_medications:
        return []
    scheduled_heads: list[str] = []
    for entry in getattr(plan, "medication_schedule", []) or []:
        raw = (
            getattr(entry, "medication", "")
            or getattr(entry, "name", "")
            or getattr(entry, "drug", "")
            or ""
        )
        head = _drug_head(raw)
        if head:
            scheduled_heads.append(head)

    omitted: list[str] = []
    seen: set[str] = set()
    for med in known_medications:
        head = _drug_head(med)
        if len(head) < 3 or head in seen:
            continue
        seen.add(head)
        found = any(head in s or s in head for s in scheduled_heads)
        if not found:
            omitted.append(head)
    return omitted


# ── Weight-bearing / surgeon-restriction gate ───────────────────────────────
# The external audit's mandate: every rehab phase must be traceable to the
# surgeon's post-op restrictions. The most common such restriction is the
# weight-bearing status (NWB < PWB < WBAT < FWB). A plan that prescribes
# exercises loading the limb BEYOND the case-stated ceiling (e.g. case = NWB,
# plan = full-weight standing/gait) is a direct patient-specific failure —
# universal and deterministic. Non-orthopaedic cases never state a WB status, so
# the gate simply never fires for them (safe across all medical fields).
_WB_RANK = {"NWB": 0, "TTWB": 1, "TDWB": 1, "PWB": 1, "WBAT": 2, "FWB": 3}
# Case-side phrase → canonical code. Abbreviations are language-agnostic; the
# spelled-out phrases cover English + Russian (the platform's clinical languages).
_WB_CASE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"\bNWB\b|non[-\s]?weight[-\s]?bearing|без\s+опоры\s+на\s+ног|без\s+нагрузки\s+на\s+ног",
        "NWB",
    ),
    (r"\bTTWB\b|\bTDWB\b|toe[-\s]?touch(?:\s+weight[-\s]?bearing)?", "PWB"),
    (r"\bPWB\b|partial[-\s]?weight[-\s]?bearing|частичн\w*\s+(?:опор|нагрузк)", "PWB"),
    (
        r"\bWBAT\b|weight[-\s]?bearing\s+as\s+tolerated|нагрузк\w*\s+по\s+переносимости",
        "WBAT",
    ),
    (r"\bFWB\b|full[-\s]?weight[-\s]?bearing|полн\w*\s+(?:осевой\s+)?нагрузк", "FWB"),
)
_WB_CASE_COMPILED = tuple(
    (re.compile(pat, re.IGNORECASE | re.UNICODE), code) for pat, code in _WB_CASE_PATTERNS
)


def _case_weight_bearing_ceiling(case_text: str) -> str | None:
    """The single weight-bearing ceiling the case commits to, or None.

    Like the operated-side gate, this fails open on ambiguity: it returns a code
    only when the case names EXACTLY ONE distinct weight-bearing level. A case
    documenting a progression ("NWB × 6 weeks, then PWB") names two → ambiguous
    which applies to the plan now → no flag.
    """
    if not case_text:
        return None
    found: set[str] = set()
    for rx, code in _WB_CASE_COMPILED:
        if rx.search(case_text):
            found.add(code)
    return next(iter(found)) if len(found) == 1 else None


def check_weight_bearing_fidelity(plan: "GeneratedPlan", case_text: str) -> dict | None:
    """Flag exercises whose weight-bearing EXCEEDS the case-stated ceiling.

    Returns a PHI-free violation dict, else ``None``:
        {"case_level": "NWB", "exceeding": ["FWB"]}
    """
    ceiling = _case_weight_bearing_ceiling(case_text or "")
    if ceiling is None:
        return None
    ceiling_rank = _WB_RANK[ceiling]
    exceeding: set[str] = set()
    for phase in getattr(plan, "phases", []) or []:
        for ex in getattr(phase, "exercises", []) or []:
            wb = (getattr(ex, "weight_bearing", "") or "").strip().upper()
            rank = _WB_RANK.get(wb)
            if rank is not None and rank > ceiling_rank:
                exceeding.add(wb)
    if exceeding:
        return {"case_level": ceiling, "exceeding": sorted(exceeding)}
    return None


# ── Monitoring-bounds sanity gate ───────────────────────────────────────────
# monitoring_parameters drive the patient's AT-HOME alerting: a reading outside
# [min_value, max_value] fires an alert. A flipped or degenerate range (min >=
# max — e.g. SBP min 180 / max 90) either alarms on every reading or on none, so
# the vital the doctor thinks is being watched is effectively mis-monitored. This
# is a deterministic, condition-agnostic consistency check (NOT clinical
# knowledge — it asserts nothing about WHAT the bound should be, only that a
# range must have min < max), so it stays within the no-hardcoded-thresholds
# rule. Unknown parameter keys are deliberately NOT flagged: only 8 keys are
# auto-evaluated, but a plan may legitimately monitor others (INR, pain), so
# flagging them would cry wolf.


def check_monitoring_bounds(plan: "GeneratedPlan") -> list[str]:
    """Canonical parameter keys whose alert range is invalid (min >= max), so the
    at-home alert can never behave as intended. Empty list when all ranges sane.
    """
    invalid: list[str] = []
    for mp in getattr(plan, "monitoring_parameters", []) or []:
        lo = getattr(mp, "min_value", None)
        hi = getattr(mp, "max_value", None)
        if lo is None or hi is None:
            continue  # a one-sided bound (only min OR only max) is legitimate
        try:
            if float(lo) >= float(hi):
                key = (getattr(mp, "parameter", "") or getattr(mp, "label", "") or "?").strip()
                invalid.append(key or "?")
        except (TypeError, ValueError):  # pragma: no cover — non-numeric bound
            continue
    return invalid


# ── Discontinued-medication-still-scheduled gate ────────────────────────────
# The audit's item-3 contradiction: the case says to STOP/HOLD a drug, but the
# plan schedules it with an active dose. Universal and deterministic. It uses the
# plan's OWN scheduled drug names as the dictionary (no external drug list), and
# flags one only when a stop/hold verb sits ADJACENT to that drug in the case
# AND the plan still doses it actively (a correctly-held drug carries its own
# STOP/HOLD marker, which is not flagged). Conservative — a false positive merely
# prompts the doctor to double-check; a false negative is the dangerous direction.
_STOP_VERB_RE = re.compile(
    r"discontinu|\bstop(?:ped|ping|s)?\b|\bhold\b|withhold|\bcease|прекра[тщ]|отмен|"
    r"to['ʻ‘’]?xtat",
    re.IGNORECASE | re.UNICODE,
)


def _case_says_stop(case_text: str, drug: str) -> bool:
    """True if a stop/hold verb appears within ~30 chars of a mention of ``drug``."""
    low = case_text.lower()
    d = drug.lower()
    idx = 0
    while True:
        i = low.find(d, idx)
        if i < 0:
            return False
        window = case_text[max(0, i - 30) : i + len(d) + 30]
        if _STOP_VERB_RE.search(window):
            return True
        idx = i + len(d)


def check_stopped_medication_scheduled(plan: "GeneratedPlan", case_text: str) -> list[str]:
    """Drug names the case says to STOP/HOLD that the plan still schedules with an
    active dose. Empty list when the plan honours every documented stop/hold.
    """
    if not case_text:
        return []
    flagged: list[str] = []
    seen: set[str] = set()
    for entry in getattr(plan, "medication_schedule", []) or []:
        name = getattr(entry, "medication", "") or getattr(entry, "name", "") or ""
        drug = _drug_head(name)
        if len(drug) < 3 or drug in seen:
            continue
        # A plan that itself marks the row STOP/HOLD is honouring the case — skip.
        marker = f"{getattr(entry, 'dose', '') or ''} {getattr(entry, 'instructions', '') or ''} {getattr(entry, 'notes', '') or ''}"
        if _STOP_VERB_RE.search(marker):
            continue
        if _case_says_stop(case_text, drug):
            seen.add(drug)
            flagged.append(drug)
    return flagged
