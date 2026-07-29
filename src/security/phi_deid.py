"""PHI de-identification for text leaving the trust boundary (e.g. the free-text
case narrative sent to an external LLM).

This is DISTINCT from `prompt_sanitizer.sanitize_for_prompt`, which is a
prompt-injection filter, NOT a de-identifier. This module removes directly
identifying tokens that a clinician may paste into a case narrative.

Design constraints (why it is conservative):
  * It must NOT corrupt clinical meaning. Blood pressure ("120/80"), rep schemes
    ("2x10"), lab values ("HbA1c 8.2"), ICD codes ("M17.0"), and doses are NOT
    identifiers and are preserved. We therefore redact only high-precision PHI
    shapes: emails, full calendar dates, long digit runs (MRN / phone / passport
    numbers), passport-style IDs, and values that follow an explicit
    name/MRN/passport LABEL (incl. RU/UZ "ФИО").
  * Free-floating full names WITHOUT a label are not reliably matchable with a
    regex and are a documented residual risk — a downstream NER pass is the
    correct place to close that (tracked as follow-up), and this module leaves a
    single, honest seam (`deidentify`) for it to hook into.

Deterministic and side-effect-free. Never raises on normal input.
"""

from __future__ import annotations

import re

EMAIL = "[EMAIL]"
DATE = "[DATE]"
IDNUM = "[ID]"
NAME = "[NAME]"

# ── high-precision patterns (order matters: labels & emails & dates first) ──

# "Name: Jane Doe", "Patient name — John Smith", "ФИО: Иванов И.И.", "Bemor: ..."
_LABELED_NAME = re.compile(
    r"(?i)\b(?:patient\s*name|full\s*name|name|ф\.?и\.?о\.?|фио|bemor|bemorning\s*ismi|ismi)\s*[:\-—]\s*"
    r"[^\n,;]{1,60}"
)
# "MRN: 12345", "passport AA1234567", "паспорт ...", "ID No: ..."
_LABELED_ID = re.compile(
    r"(?i)\b(?:mrn|medical\s*record(?:\s*(?:no|number|#))?|passport|паспорт|pasport|"
    r"id\s*(?:no|number|#)?|record\s*(?:no|number|#))\s*[:\-—]?\s*[A-Za-z]{0,2}[\s-]?\d[\d\s-]{2,}"
)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Full calendar dates: DD/MM/YYYY, DD.MM.YYYY, DD-MM-YYYY, YYYY-MM-DD (and 2-digit year).
_DATE_NUMERIC = re.compile(
    r"\b("
    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"  # 01/02/1950
    r"|\d{4}[./-]\d{1,2}[./-]\d{1,2}"  # 1950-02-01
    r")\b"
)
# "1 Jan 1950", "Jan 1, 1950", "01 January 1950" (EN month names — cheap, high-precision).
_DATE_ALPHA = re.compile(
    r"(?i)\b("
    r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{2,4}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}"
    r")\b"
)
# Passport-style: 1-2 letters + 6-9 digits (e.g. UZ "AA1234567").
_PASSPORT = re.compile(r"\b[A-Za-z]{1,2}\d{6,9}\b")
# Any run of ≥7 digits (optionally +/separators) → phone / MRN / passport number.
# 7-digit floor keeps clinical values safe: BP (3), reps (2), most labs (<5).
_LONG_DIGITS = re.compile(r"(?<![\w])\+?\d[\d\s().-]{5,}\d(?![\w])")


def _digit_count(s: str) -> int:
    return sum(ch.isdigit() for ch in s)


def _redact_long_digits(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        # Only redact if it truly carries ≥7 digits — avoids eating "120/80" or
        # spaced dose text that happens to match the loose separator class.
        return IDNUM if _digit_count(m.group(0)) >= 7 else m.group(0)

    return _LONG_DIGITS.sub(repl, text)


def deidentify(text: str | None) -> str:
    """Return ``text`` with high-precision direct identifiers redacted.

    Safe on ``None``/empty (returns ""). Never raises on normal input.
    """
    if not text:
        return ""
    try:
        out = _LABELED_NAME.sub(lambda m: _label_prefix(m.group(0)) + NAME, text)
        out = _LABELED_ID.sub(lambda m: _label_prefix(m.group(0)) + IDNUM, out)
        out = _EMAIL.sub(EMAIL, out)
        out = _DATE_NUMERIC.sub(DATE, out)
        out = _DATE_ALPHA.sub(DATE, out)
        out = _PASSPORT.sub(IDNUM, out)
        out = _redact_long_digits(out)
        return out
    except Exception:  # noqa: BLE001 — de-id must fail SAFE, never emit raw PHI
        # Fail-closed for PHI: if anything unexpected happens, drop the narrative
        # rather than leak it. Generation still proceeds on structured fields.
        return "[REDACTED]"


def _label_prefix(matched: str) -> str:
    """Keep the label keyword (clinically useful context) but drop its value.

    "Name: John Smith" -> "Name: " ; "MRN: 12345" -> "MRN: ".
    """
    m = re.match(r"(?i)\s*([^:\-—]*[:\-—]\s*)", matched)
    return m.group(1) if m else ""
