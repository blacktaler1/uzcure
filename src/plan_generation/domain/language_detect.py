"""Field-level English-leak detection for UZ/RU plans.

The script-based `looks_compliant` cannot tell Latin-script Uzbek from English
(both are Latin). When the LLM is asked for an Uzbek plan it sometimes copies an
English drug fact verbatim into a free-text field (e.g. a `safety_alert`). This
module flags such fields by a conservative marker heuristic so the use case can
trigger a retry — never a hard block (the plan always reaches the doctor).
"""

from __future__ import annotations

import re
from typing import Any

# High-frequency English function / medical words rare in Uzbek free text.
_EN_MARKERS = frozenset(
    {
        "the",
        "and",
        "or",
        "of",
        "in",
        "to",
        "with",
        "for",
        "is",
        "are",
        "be",
        "no",
        "not",
        "if",
        "this",
        "from",
        "at",
        "per",
        "each",
        "when",
        "while",
        "without",
        "risk",
        "increased",
        "decreased",
        "monitor",
        "monitoring",
        "exercise",
        "blood",
        "heart",
        "rate",
        "avoid",
        "during",
        "after",
        "before",
        "high",
        "low",
        "dose",
        "daily",
        "patient",
        "due",
        "may",
        "should",
        "check",
        "means",
        "have",
        "available",
        "recurrent",
        "delayed",
        "fall",
        "supervised",
        "balance",
        "gait",
        "position",
        "changes",
        "additive",
        "severe",
        "sedation",
        "depression",
        "mandatory",
        "hydration",
        "combination",
        "level",
        "levels",
        "effect",
        "effects",
    }
)

# Uzbek high-frequency markers (Latin script). Apostrophe variants are
# normalized to ASCII "'" before matching so "boʻlsa"/"bo'lsa" both count.
_UZ_MARKERS = frozenset(
    {
        "va",
        "bilan",
        "uchun",
        "kerak",
        "yoki",
        "ham",
        "bu",
        "har",
        "agar",
        "mashq",
        "mashqlar",
        "dori",
        "dorilar",
        "xavf",
        "xavfi",
        "nazorat",
        "yurak",
        "qon",
        "bosim",
        "oshgan",
        "kamaygan",
        "davomida",
        "keyin",
        "oldin",
        "yuqori",
        "past",
        "ichida",
        "shart",
        "bemor",
        "holat",
        "emas",
        "sababli",
        "hamda",
        "yo'q",
        "bo'lsa",
        "bo'yicha",
        "ko'rik",
        "ko'rsatma",
        "tavsiya",
        "tekshirilsin",
        "qilinadi",
        "kuni",
        "kun",
        "tushirilsin",
        "to'xtatilsin",
        "qabul",
    }
)

_WORD_RE = re.compile(r"[a-zA-Z']+")
_APOSTROPHES = {"ʻ": "'", "ʼ": "'", "‘": "'", "’": "'"}


def _normalize_apostrophes(text: str) -> str:
    for src, dst in _APOSTROPHES.items():
        text = text.replace(src, dst)
    return text


# Value keys whose strings are identifiers / codes / URLs — never free text.
_SKIP_KEYS = frozenset(
    {
        "exercise_id",
        "video_url",
        "source",
        "severity",
        "patient_name",
        "id",
        "engine_version",
        "plan_status",
        "provider_used",
        "start_date",
        "reward_day",
        "specialty_detected",
        "decision",
        "verdict",
    }
)


def is_probably_english(text: str) -> bool:
    """Conservative: a Latin free-text string that reads as English, not Uzbek.
    Needs several English markers AND more English than Uzbek markers, so brand
    names and abbreviations do not false-trip."""
    if not text:
        return False
    words = [w.lower() for w in _WORD_RE.findall(_normalize_apostrophes(text))]
    if len(words) < 4:
        return False
    en = sum(1 for w in words if w in _EN_MARKERS)
    uz = sum(1 for w in words if w in _UZ_MARKERS)
    return en >= 2 and en > uz


def collect_human_text(
    data: Any, key: str | None = None, out: list[str] | None = None
) -> list[str]:
    """Walk a plan dict and collect multi-word free-text values (skips codes,
    ids, URLs). Used to run `is_probably_english` per field."""
    if out is None:
        out = []
    if isinstance(data, dict):
        for k, v in data.items():
            collect_human_text(v, k, out)
    elif isinstance(data, list):
        for v in data:
            collect_human_text(v, key, out)
    elif isinstance(data, str):
        if key in _SKIP_KEYS:
            return out
        s = data.strip()
        if len(s) >= 8 and " " in s:
            out.append(s)
    return out
