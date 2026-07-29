"""Contraindication scan — short-clinical-term coverage (audit fix C-1a).

Root defect: `_tokenize` drops tokens <4 chars, so safety-critical short terms
("hip", "ACL", "DVT", "ORIF", "TKR", …) were silently discarded from BOTH the
diagnosis and the contraindication string — a "hip precautions" contraindication
could never match a "hip replacement" diagnosis. These tests pin the fix and
guard against regression, in both directions (new matches surface; unrelated
diagnoses still produce NO false match).
"""

from __future__ import annotations

from app.features.plan_generation.application.safety_check import (
    SafetyCheckResult,
    _scan_contraindications,
)
from app.features.plan_generation.domain import Exercise


def _exercise() -> Exercise:
    return Exercise(
        exercise_id="EX1",
        name="Test Exercise",
        sets=3,
        reps=10,
    )


class _Registry:
    def __init__(self, contraindications: list[str]) -> None:
        self.contraindications = contraindications


def _scan(diagnosis: str, contraindication: str) -> SafetyCheckResult:
    result = SafetyCheckResult()
    _scan_contraindications(
        _exercise(),
        _Registry([contraindication]),  # type: ignore[arg-type]
        [diagnosis.lower()],
        result,
    )
    return result


# ── Short-term matches that were previously MISSED (the bug) ────────────────

def test_hip_short_term_now_matches():
    r = _scan("posterior hip replacement precautions", "hip flexion beyond 90 degrees")
    assert len(r.contraindications) == 1
    assert r.contraindications[0].exercise_id == "EX1"


def test_dvt_short_term_now_matches():
    r = _scan("acute DVT left calf", "active DVT or recent VTE")
    assert len(r.contraindications) == 1


def test_acl_short_term_now_matches():
    r = _scan("ACL reconstruction week 2", "ACL graft protection phase")
    assert len(r.contraindications) == 1


def test_orif_short_term_now_matches():
    r = _scan("distal radius ORIF", "ORIF hardware not consolidated")
    assert len(r.contraindications) == 1


# ── Existing ≥4-char behavior must still work (regression guard) ────────────

def test_long_token_overlap_still_matches():
    r = _scan("post-op knee fracture", "unstable knee fracture")
    assert len(r.contraindications) == 1


# ── No false positives on unrelated diagnoses (the other direction) ─────────

def test_unrelated_diagnosis_no_false_match():
    r = _scan("shoulder impingement syndrome", "unstable knee fracture")
    assert r.contraindications == []


def test_short_term_not_present_no_false_match():
    # "hip" contraindication, but diagnosis is about the shoulder — must NOT match.
    r = _scan("adhesive capsulitis of the shoulder", "hip precautions post arthroplasty")
    assert r.contraindications == []


def test_empty_inputs_are_safe():
    assert _scan("", "hip precautions").contraindications == []
    assert _scan("hip replacement", "").contraindications == []
