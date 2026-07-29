"""
Secure Logging Utility
======================
E-10: Masks PHI/PII in log messages to ensure HIPAA compliance.
"""

import logging
import re
from collections.abc import Callable
from re import Match
from typing import cast


def _luhn_valid(digits: str) -> bool:
    """Return True for likely payment-card PANs; avoids masking random long IDs."""
    if not digits.isdigit():
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        n = ord(ch) - ord("0")
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _mask_pan(match: Match[str]) -> str:
    """Mask payment-card numbers, preserving only the last four digits."""
    raw = match.group()
    digits = re.sub(r"\D", "", raw)
    if not (13 <= len(digits) <= 19) or not _luhn_valid(digits):
        return raw
    return "****" + digits[-4:]


def _mask_patient_name_phrase(match: Match[str]) -> str:
    """Mask free-text names after patient/bemor labels."""
    return f"{match.group(1)} ***"


class PHIMaskingFilter(logging.Filter):
    """
    Logging filter that masks potential PHI/PII data.
    """

    # Patterns to mask
    PATTERNS = [
        # Payment cards / PAN (13-19 digits, optional spaces/dashes, Luhn-checked).
        (r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)", _mask_pan),
        # US-style Social Security numbers.
        (r"\b\d{3}-\d{2}-\d{4}\b", "***-**-****"),
        # Keyed medical record numbers / IDs.
        (
            r"\b(MRN|medical_record(?:_number)?|record_number)\s*[:=]\s*([A-Za-z0-9][A-Za-z0-9_.-]{2,63})\b",
            r"\1=***",
        ),
        # Keyed dates of birth.
        (
            r"\b(DOB|date_of_birth|birth_date|birthdate)\s*[:=]\s*(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
            r"\1=***",
        ),
        # Keyed street/address fields.
        (r"\b(address|addr)\s*[:=]\s*([^;\n]+)", r"\1=***"),
        # Keyed clinical free-text (diagnosis / complaint / allergen). Mirrors the
        # MRN/DOB/address keyed patterns: only fires after an explicit clinical
        # label + ':'/'=' (e.g. "diagnosis: Acute MI"), so operational logs that
        # merely mention the word (no value) are untouched. Defense-in-depth — the
        # code is disciplined about not logging these; this catches a future slip.
        (
            r"\b(diagnosis|primary_diagnosis|secondary_diagnoses|chief_complaint|complaint|allergen)\s*[:=]\s*([^;\n]+)",
            r"\1=***",
        ),
        # Free-text names after common patient labels, e.g. "Patient John Smith".
        (
            r"\b(Patient|patient|Bemor|bemor)\s+([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,3})\b",
            _mask_patient_name_phrase,
        ),
        # Phone numbers (various formats)
        (r"\+?[0-9]{10,15}", lambda m: m.group()[:3] + "***" + m.group()[-2:]),
        # Email addresses
        (
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            lambda m: m.group()[:2] + "***@***",
        ),
        # Patient names (common patterns)
        (r'patient[_\s]?name["\s:=]+["\']?([^"\'&\n]+)["\']?', r"patient_name=***"),
        # Full names after common keywords
        (r'(name|full_name)["\s:=]+["\']?([A-Z][a-z]+ [A-Z][a-z]+)["\']?', r"\1=***"),
        # URL query secrets (api_key, token, password, etc.)
        (
            r'([?&](?:api_key|apikey|token|access_token|authorization|key|secret|password)=)([^&\s"\'\\]+)',
            r"\1***",
        ),
        # Plain key=value secrets in formatted log messages
        (
            r'((?:^|[\s,;])(?:api_key|apikey|token|access_token|authorization|key|secret|password)=)([^\s,;"\']+)',
            r"\1***",
        ),
        # JSON-style secret fields
        (
            r'("(?:api_key|apikey|token|access_token|authorization|secret|password)"\s*:\s*")([^"]+)(")',
            r"\1***\3",
        ),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Mask sensitive data in log message."""
        # BUG-LOG-1 FIX: Uvicorn access logger requires args for its AccessFormatter.
        # If we clear them, AccessFormatter crashes with ValueError (expected 5, got 0).
        is_uvicorn_access = record.name == "uvicorn.access"

        if hasattr(record, "args") and record.args:
            if is_uvicorn_access:
                # For uvicorn access, we mask individual args if they are strings,
                # but we MUST preserve the tuple structure/length.
                new_args = []
                for arg in record.args:
                    if isinstance(arg, str):
                        new_args.append(self.mask_sensitive(arg))
                    else:
                        new_args.append(arg)
                record.args = tuple(new_args)
                # Also mask the raw message template if it's a string
                if isinstance(record.msg, str):
                    record.msg = self.mask_sensitive(record.msg)
            else:
                # Render first so placeholder-based messages ("token=%s")
                # are masked after interpolation, then clear args to avoid
                # double-formatting in logging internals.
                try:
                    rendered = record.getMessage()
                except Exception:
                    rendered = str(record.msg)
                record.msg = self.mask_sensitive(rendered)
                record.args = ()
        elif hasattr(record, "msg") and record.msg:
            record.msg = self.mask_sensitive(str(record.msg))
        return True

    def mask_sensitive(self, text: str) -> str:
        """Apply all masking patterns to text."""
        for pattern, replacement in self.PATTERNS:
            if callable(replacement):
                # Ensure the callable is treated as (Match[str]) -> str
                repl_func = cast(Callable[[Match[str]], str], replacement)
                text = re.sub(pattern, repl_func, text)
            else:
                text = re.sub(pattern, cast(str, replacement), text, flags=re.IGNORECASE)
        return text


def mask_phone(phone: str) -> str:
    """Mask phone number for logging. Shows first 3 and last 2 digits."""
    if not phone:
        return "***"
    clean = re.sub(r"\D", "", phone)
    if len(clean) < 5:
        return "***"
    return clean[:3] + "***" + clean[-2:]


def mask_name(name: str) -> str:
    """Mask name for logging. Shows first initial only."""
    if not name:
        return "***"
    parts = name.split()
    return " ".join(p[0] + "***" for p in parts if p)


def safe_log_patient(patient_id: int, action: str) -> str:
    """Create safe log message for patient actions."""
    return f"Patient #{patient_id}: {action}"


def setup_secure_logging():
    """Add PHI masking filter to all handlers."""
    phi_filter = PHIMaskingFilter()
    for handler in logging.root.handlers:
        handler.addFilter(phi_filter)

    # Also add to named loggers
    for name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            handler.addFilter(phi_filter)


# Example usage in other modules:
# from app.core.secure_logging import mask_phone, safe_log_patient
# logger.info(f"User logged in: {mask_phone(user.phone)}")
# logger.info(safe_log_patient(patient.id, "completed exercise"))
