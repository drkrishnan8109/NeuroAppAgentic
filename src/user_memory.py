"""Fail-open, user-scoped adapter around a Mem0-compatible store."""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable
from typing import Any

from src.memory_schema import (
    CATEGORIES,
    SCHEMA_VERSION,
    CandidatePreference,
    should_extract_preferences,
    validate_candidate,
)


logger = logging.getLogger(__name__)
PreferenceMap = dict[str, list[str]]
Extractor = Callable[[str], Iterable[CandidatePreference]]


def build_subject_hash(subject: str) -> str:
    """Provider-independent fingerprint used only to support future re-keying."""
    if not subject or not subject.strip():
        raise ValueError("OIDC subject is required")
    return hashlib.sha256(subject.strip().encode("utf-8")).hexdigest()


def format_preferences(preferences: PreferenceMap) -> str:
    """Render a bounded style block; callers must fence it before prompting an LLM."""
    lines: list[str] = []
    for category in CATEGORIES:
        for fact in (preferences.get(category) or [])[:2]:
            candidate = validate_candidate(CandidatePreference(fact=fact, category=category))
            if candidate:
                lines.append(f"- {category}: {candidate.fact}")
    return "\n".join(lines)


class UserMemory:
    def __init__(self, store: Any, extractor: Extractor | None = None):
        self._store = store
        self._extractor = extractor

    def load_preferences(self, user_id: str) -> PreferenceMap:
        """Load at most two validated facts per category; an outage returns no prefs."""
        try:
            response = self._store.get_all(filters={"user_id": user_id}, top_k=100)
            rows = response.get("results", []) if isinstance(response, dict) else response
        except Exception as exc:
            logger.warning("memory read failed: %s", type(exc).__name__)
            return {}

        grouped: PreferenceMap = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            category = row.get("category") or metadata.get("category")
            fact = row.get("memory") or row.get("data") or ""
            candidate = validate_candidate(
                CandidatePreference(fact=str(fact), category=str(category or ""))
            )
            if not candidate:
                continue
            bucket = grouped.setdefault(candidate.category, [])
            if candidate.fact not in bucket and len(bucket) < 2:
                bucket.append(candidate.fact)
        return grouped

    def remember_from_text(
        self, text: str, user_id: str, subject_hash: str
    ) -> dict[str, int]:
        """Extract locally, validate, then direct-import accepted facts into Mem0."""
        report = {"extracted": 0, "stored": 0, "dropped": 0}
        if not should_extract_preferences(text) or self._extractor is None:
            return report

        try:
            candidates = list(self._extractor(text))
        except Exception as exc:
            logger.warning("preference extraction failed: %s", type(exc).__name__)
            return report

        report["extracted"] = len(candidates)
        existing = {
            fact for facts in self.load_preferences(user_id).values() for fact in facts
        }
        for raw in candidates:
            candidate = validate_candidate(raw)
            if not candidate:
                report["dropped"] += 1
                continue
            if candidate.fact in existing:
                continue
            try:
                self._store.add(
                    candidate.fact,
                    user_id=user_id,
                    metadata={
                        "category": candidate.category,
                        "schema_version": SCHEMA_VERSION,
                        "subject_hash": subject_hash,
                    },
                    infer=False,
                )
                report["stored"] += 1
                existing.add(candidate.fact)
            except Exception as exc:
                logger.warning(
                    "memory write failed category=%s error=%s",
                    candidate.category,
                    type(exc).__name__,
                )
        logger.info("memory extraction counts=%s", report)
        return report

    def delete_all(self, user_id: str) -> None:
        """Delete only the authenticated user's partition."""
        if not user_id:
            raise ValueError("user_id is required")
        self._store.delete_all(user_id=user_id)
