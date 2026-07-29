"""Language compliance check for LLM output.

Cheap heuristic — only fails when the dominant script is clearly the wrong
one. Tolerates mixed-script content (medical abbreviations like BNP/NYHA,
ICD codes, brand names, patient identifiers) that legitimately appears in
all three target languages.
"""

from __future__ import annotations

import re

from app.features.plan_generation.domain import PlanLanguage

CYRILLIC_RANGE = (0x0400, 0x04FF)

# Cyrillic letters used by RUSSIAN but NOT by the Uzbek Cyrillic alphabet.
_RUSSIAN_ONLY_CYR = frozenset("ыэщъё")
# Cyrillic letters unique to UZBEK Cyrillic (never in Russian).
_UZBEK_ONLY_CYR = frozenset("ўқғҳ")
# Uzbek-Latin apostrophe-letter markers (ASCII and typographic tutuq belgisi).
_UZBEK_LATIN_MARKERS = ("o'", "g'", "oʻ", "gʻ", "o‘", "g‘")
# Common Uzbek function/clinical words — presence is a strong "this is Uzbek" signal.
_UZBEK_WORDS = frozenset(
    {
        "va",
        "bilan",
        "uchun",
        "kun",
        "kunlik",
        "kunda",
        "hafta",
        "haftalik",
        "mashq",
        "mashqlar",
        "takror",
        "takrorlang",
        "marta",
        "bemor",
        "bemorga",
        "zaruriy",
        "davolash",
        "nafas",
        "olish",
        "kerak",
        "quyidagi",
    }
)
# Common English function words — several together (with no Uzbek signal) mean
# the "Uzbek" output is really English.
_ENGLISH_WORDS = frozenset(
    {
        "the",
        "and",
        "with",
        "for",
        "daily",
        "times",
        "week",
        "weeks",
        "day",
        "days",
        "perform",
        "patient",
        "exercise",
        "exercises",
        "reps",
        "sets",
        "should",
        "your",
        "this",
        "these",
        "hold",
        "seconds",
        "repeat",
    }
)


def looks_compliant(text: str, language: PlanLanguage) -> bool:
    if not text:
        return False

    cyr = sum(1 for ch in text if CYRILLIC_RANGE[0] <= ord(ch) <= CYRILLIC_RANGE[1])
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    total = cyr + latin
    if total == 0:
        return True
    cyr_ratio = cyr / total

    if language is PlanLanguage.RU:
        # Russian must be majority Cyrillic. Lowered from 0.40 — medical
        # abbreviations (BNP, NYHA II, EF<50%, ICD I60), units, drug brand
        # names sit in Latin and a strict 40% bar fails legitimate output.
        if cyr_ratio < 0.20:
            return False
        # Cyrillic alone cannot tell Russian from Uzbek-Cyrillic (shared script).
        # A reply carrying Uzbek-only Cyrillic letters (ў/қ/ғ/ҳ) is Uzbek-Cyrillic,
        # not Russian — reject so the retry can produce actual Russian. Real
        # Russian text does not use these letters.
        lower = text.lower()
        has_uz_cyr = any(ch in _UZBEK_ONLY_CYR for ch in lower)
        return not has_uz_cyr

    if language is PlanLanguage.UZ_CYRILLIC:
        # We explicitly requested the Cyrillic script, so the output must be
        # Cyrillic-dominant. Same tolerant 0.20 floor as Russian — Latin medical
        # abbreviations (NYHA, BNP), units and drug brands legitimately sit in
        # Latin. A Latin-dominant reply means the model ignored the script
        # request and is treated as non-compliant (triggers a retry, never a
        # hard fail).
        if cyr_ratio < 0.20:
            return False
        # Cyrillic alone cannot tell Uzbek-Cyrillic from Russian (shared script) —
        # the previous ratio-only check accepted a plain RUSSIAN reply as Uzbek-
        # Cyrillic, so an Uzbek-Cyrillic patient could receive a Russian plan.
        # Mirror the UZ branch: a reply using Russian-only letters (ы/э/щ/ъ/ё)
        # with NO Uzbek-Cyrillic letters (ў/қ/ғ/ҳ) is Russian — reject it.
        lower = text.lower()
        has_uz_cyr = any(ch in _UZBEK_ONLY_CYR for ch in lower)
        has_ru_only = any(ch in _RUSSIAN_ONLY_CYR for ch in lower)
        return not (has_ru_only and not has_uz_cyr)

    if language is PlanLanguage.UZ:
        # Uzbek is written in Latin OR Cyrillic — accept both. But UZ is the
        # DEFAULT language, so an unconditional pass (the previous behaviour)
        # meant an English- or Russian-labelled-UZ plan reached Uzbek patients
        # unchecked. Reject only HIGH-CONFIDENCE wrong-language (a false reject
        # would block a legitimate plan):
        lower = text.lower()
        has_uz_cyr = any(ch in _UZBEK_ONLY_CYR for ch in lower)
        if cyr_ratio >= 0.5:
            # Cyrillic-dominant. If it uses Russian-only letters and NO
            # Uzbek-Cyrillic letters, it is Russian — reject.
            has_ru_only = any(ch in _RUSSIAN_ONLY_CYR for ch in lower)
            return not (has_ru_only and not has_uz_cyr)
        # Latin-dominant. It is English only if it carries several English
        # function words AND shows zero Uzbek signal (markers or words).
        tokens = set(re.findall(r"[a-z']+", lower))
        has_uz_signal = bool(tokens & _UZBEK_WORDS) or any(m in lower for m in _UZBEK_LATIN_MARKERS)
        english_hits = len(tokens & _ENGLISH_WORDS)
        if english_hits >= 3 and not has_uz_signal:
            return False
        return True

    if language is PlanLanguage.EN:
        # English must be majority Latin. Some Cyrillic patient names or
        # comorbidities legitimately appear in EN output for CIS clinics —
        # accept up to half before flagging a mismatch.
        return cyr_ratio < 0.50

    return True
