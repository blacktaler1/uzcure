"""Magic-byte signature validation for uploaded files.

P1-004 remediation. Pure Python implementation — no system libraries
required. Checks the first bytes of the file against a whitelist of
known signatures. Does not depend on libmagic.

Whitelist is intentionally narrow: only the file types our upload
pipeline accepts (PDF, JPEG, PNG, DOCX, plain text, DOC).
"""

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)


# Map declared content-type -> tuple of acceptable byte prefixes.
# Any prefix matches means the file passes signature validation.
_SIGNATURES: Final[dict[str, tuple[bytes, ...]]] = {
    "application/pdf": (b"%PDF-",),
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (b"PK\x03\x04",),
    "application/msword": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
    # text/plain is special — handled via _is_likely_text below.
}

# Bytes we read for the signature check. 32 covers every prefix above.
_SNIFF_BYTES: Final[int] = 32


class SignatureValidationError(ValueError):
    def __init__(self, *, declared: str, actual: str, filename: str):
        self.declared = declared
        self.actual = actual
        self.filename = filename
        super().__init__(
            f"Signature mismatch: declared={declared} actual={actual} filename={filename}"
        )


def _is_likely_text(data: bytes) -> bool:
    """True if data is valid UTF-8 with no null bytes."""
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _identify_actual(head: bytes) -> str:
    """Best-effort identification of what the file actually is.

    Used only for logging and for the SignatureValidationError.actual
    field. Never trusted for routing decisions.
    """
    for mime, prefixes in _SIGNATURES.items():
        if any(head.startswith(p) for p in prefixes):
            return mime
    if _is_likely_text(head):
        return "text/plain"
    return "unknown"


def validate_file_signature(
    *,
    declared_content_type: str,
    file_bytes: bytes,
    filename: str = "<unknown>",
) -> None:
    """Raise SignatureValidationError if file bytes don't match declared type."""
    if not file_bytes:
        logger.warning(
            "Upload rejected: empty file filename=%s declared=%s",
            filename,
            declared_content_type,
        )
        raise SignatureValidationError(
            declared=declared_content_type,
            actual="empty",
            filename=filename,
        )

    head = file_bytes[:_SNIFF_BYTES]

    # text/plain has no magic bytes — fall back to UTF-8 heuristic.
    if declared_content_type == "text/plain":
        if _is_likely_text(head):
            logger.debug("Signature validated (text): filename=%s", filename)
            return
        actual = _identify_actual(head)
        logger.warning(
            "Upload rejected: declared text but binary content filename=%s actual=%s",
            filename,
            actual,
        )
        raise SignatureValidationError(
            declared="text/plain",
            actual=actual,
            filename=filename,
        )

    sigs = _SIGNATURES.get(declared_content_type)
    if not sigs:
        logger.warning(
            "Upload rejected: unsupported declared content type filename=%s declared=%s",
            filename,
            declared_content_type,
        )
        raise SignatureValidationError(
            declared=declared_content_type,
            actual="unsupported_declared_type",
            filename=filename,
        )

    if not any(head.startswith(s) for s in sigs):
        actual = _identify_actual(head)
        logger.warning(
            "Upload rejected: signature mismatch filename=%s declared=%s actual=%s",
            filename,
            declared_content_type,
            actual,
        )
        raise SignatureValidationError(
            declared=declared_content_type,
            actual=actual,
            filename=filename,
        )

    logger.debug(
        "Signature validated: filename=%s declared=%s",
        filename,
        declared_content_type,
    )
