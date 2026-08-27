"""Closed preference taxonomy and deterministic pre-storage controls.

The extraction model is advisory.  Only candidates accepted here may cross the
Mem0 storage boundary.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Final


SCHEMA_VERSION: Final[str] = "1"
CATEGORIES: Final[dict[str, str]] = {
    "answer_style": "Answer shape, organization, and length.",
    "citation_depth": "How much source and page detail to show.",
    "source_preference": "Which indexed reference texts to prefer.",
    "unit_convention": "How quantities and doses should be expressed.",
    "terminology_level": "Desired jargon and abbreviation level.",
    "clinical_focus": "The clinician's standing subspecialty or care setting.",
}

CUSTOM_INSTRUCTIONS: Final[str] = """
Extract ONLY standing preferences of the clinician using this assistant.
Return JSON matching this shape exactly:
{"preferences": [{"fact": "short preference", "category": "one allowed category"}]}

Allowed categories: answer_style, citation_depth, source_preference,
unit_convention, terminology_level, clinical_focus.

NEVER extract patient details of any kind (age, sex, name, MRN, dates,
presenting complaint, case narrative), the clinical question itself, the
assistant's answer, or retrieved source text. A message describing a patient
contains no extractable preferences.

Input: Can you keep these shorter? Bullets are fine, and I always want the page number.
Output: {"preferences": [{"fact": "prefers short bulleted answers", "category": "answer_style"}, {"fact": "always wants page numbers", "category": "citation_depth"}]}

Input: 54F with status epilepticus refractory to lorazepam, what's next?
Output: {"preferences": []}

Input: I'm in the neuro ICU so assume ventilated patients.
Output: {"preferences": [{"fact": "works in neuro ICU", "category": "clinical_focus"}]}
""".strip()


@dataclass(frozen=True)
class CandidatePreference:
    fact: str
    category: str


_PREFERENCE_CUE = re.compile(
    r"\b(?:i\s+(?:always\s+|usually\s+)?(?:prefer|want|like|work|am|need)|"
    r"please\s+(?:keep|use|include|show|avoid|expand)|"
    r"for\s+me[, ]|my\s+(?:preference|specialty|practice|work))\b",
    re.IGNORECASE,
)
_AGE = re.compile(r"\b(?:\d{1,3}\s*[- ]?(?:y/?o|years? old)|\d{1,3}[MF])\b", re.IGNORECASE)
_DATE = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
)
_MRN = re.compile(r"\b(?:MRN|medical record(?: number)?)\s*[:#-]?\s*[A-Z0-9-]+\b", re.IGNORECASE)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PERSON_NAME = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Patient)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b")
_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget)\b.{0,40}\b(?:instructions?|prompt|rules?)\b|"
    r"\b(?:system|assistant)\s*:|<\|im_(?:start|end)\|>|\[/?INST\]",
    re.IGNORECASE,
)


def build_user_id(subject: str, pepper: str) -> str:
    """Build a stable pseudonymous key from the OIDC subject claim."""
    if not subject or not subject.strip():
        raise ValueError("OIDC subject is required")
    if not pepper:
        raise ValueError("MEMORY_USER_PEPPER is required")
    digest = hashlib.sha256(f"{pepper}:{subject.strip()}".encode("utf-8")).hexdigest()
    return digest[:32]


def should_extract_preferences(text: str) -> bool:
    """Cheap gate that avoids an extraction call for normal clinical questions."""
    value = (text or "").strip()
    patient_shaped = any(
        pattern.search(value)
        for pattern in (_AGE, _DATE, _MRN, _EMAIL, _SSN, _CARD, _IP, _PERSON_NAME)
    )
    return bool(
        value
        and len(value) <= 4000
        and not patient_shaped
        and _PREFERENCE_CUE.search(value)
    )


def validate_candidate(candidate: CandidatePreference) -> CandidatePreference | None:
    """Return a normalized safe candidate, or ``None`` when it must be dropped."""
    category = (candidate.category or "").strip()
    fact = " ".join((candidate.fact or "").split())
    if category not in CATEGORIES or not fact or len(fact) > 120:
        return None
    if any(
        pattern.search(fact)
        for pattern in (
            _AGE,
            _DATE,
            _MRN,
            _EMAIL,
            _SSN,
            _CARD,
            _IP,
            _PERSON_NAME,
            _INJECTION,
        )
    ):
        return None
    return CandidatePreference(fact=fact, category=category)
