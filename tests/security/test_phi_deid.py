"""PHI de-identifier (audit fix C-3).

Proves BOTH directions: direct identifiers are redacted, AND clinical content
(BP, labs, doses, ICD codes, diagnosis prose) survives unchanged.
"""

from __future__ import annotations

from app.core.phi_deid import deidentify


# ── identifiers ARE removed ─────────────────────────────────────────────────

def test_email_redacted():
    assert "jane@example.com" not in deidentify("contact jane@example.com for records")


def test_uz_phone_redacted():
    out = deidentify("phone +998901234567, call anytime")
    assert "998901234567" not in out
    assert "901234567" not in out


def test_numeric_dob_redacted():
    for raw in ("DOB 01/02/1950", "born 1950-02-01", "d.o.b 1.2.50"):
        assert not any(c.isdigit() for c in deidentify(raw).replace("[DATE]", "")) or "195" not in deidentify(raw)


def test_alpha_date_redacted():
    assert "1950" not in deidentify("date of birth 1 Jan 1950")
    assert "1950" not in deidentify("born Jan 1, 1950")


def test_mrn_label_redacted():
    out = deidentify("MRN: 4456123 admitted today")
    assert "4456123" not in out


def test_passport_redacted():
    out = deidentify("passport AA1234567 on file")
    assert "AA1234567" not in out
    assert "1234567" not in out


def test_labeled_name_redacted():
    out = deidentify("Patient name: John Smith, 60M")
    assert "John Smith" not in out
    out_ru = deidentify("ФИО: Иванов Иван Иванович")
    assert "Иванов" not in out_ru


def test_long_id_run_redacted():
    assert "123456789" not in deidentify("record 123456789 in system")


# ── clinical content SURVIVES (must not corrupt meaning) ────────────────────

def test_blood_pressure_survives():
    out = deidentify("BP 120/80, HR 72")
    assert "120/80" in out
    assert "72" in out


def test_lab_values_survive():
    out = deidentify("HbA1c 8.2%, creatinine 1.1, eGFR 65")
    assert "8.2" in out and "1.1" in out and "65" in out


def test_dose_scheme_survives():
    out = deidentify("prescribe 2x10 reps, 3 sets daily")
    assert "2x10" in out and "3 sets" in out


def test_icd_code_survives():
    out = deidentify("diagnosis M17.0 bilateral knee OA")
    assert "M17.0" in out
    assert "knee OA" in out


def test_diagnosis_prose_survives():
    text = "post-op ACL reconstruction week 2, guarded weight-bearing, mild effusion"
    assert deidentify(text) == text


# ── safety / robustness ─────────────────────────────────────────────────────

def test_none_and_empty_safe():
    assert deidentify(None) == ""
    assert deidentify("") == ""
