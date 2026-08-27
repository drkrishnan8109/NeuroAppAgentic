"""Runtime factories for structured preference extraction and Mem0 OSS."""
from __future__ import annotations

from typing import Any, TypedDict

from src.memory_schema import CUSTOM_INSTRUCTIONS, CandidatePreference


class _PreferenceRow(TypedDict):
    fact: str
    category: str


class _PreferenceExtraction(TypedDict):
    preferences: list[_PreferenceRow]


def make_preference_extractor(llm):
    """Build a local structured extractor; returned data is still untrusted."""
    structured = llm.with_structured_output(_PreferenceExtraction)

    def extract(text: str) -> list[CandidatePreference]:
        result = structured.invoke(
            [
                {"role": "system", "content": CUSTOM_INSTRUCTIONS},
                {"role": "user", "content": text},
            ]
        )
        rows = result.get("preferences", []) if isinstance(result, dict) else []
        candidates: list[CandidatePreference] = []
        for row in rows:
            if not isinstance(row, dict) or "fact" not in row or "category" not in row:
                continue
            candidates.append(
                CandidatePreference(fact=str(row["fact"]), category=str(row["category"]))
            )
        return candidates

    return extract


def build_mem0_config(connection_string: str, google_api_key: str) -> dict[str, Any]:
    """Build the documented Gemini-embedding + Supabase OSS configuration."""
    if not connection_string or not google_api_key:
        raise ValueError("SUPABASE_DB_URL and GOOGLE_API_KEY are required")
    return {
        "llm": {
            "provider": "gemini",
            "config": {
                "model": "gemini-2.0-flash-lite-001",
                "api_key": google_api_key,
                "temperature": 0.0,
            },
        },
        "embedder": {
            "provider": "gemini",
            "config": {
                "model": "models/gemini-embedding-001",
                "api_key": google_api_key,
                "embedding_dims": 768,
                "output_dimensionality": 768,
            },
        },
        "vector_store": {
            "provider": "supabase",
            "config": {
                "connection_string": connection_string,
                "collection_name": "neuro_user_preferences",
                "embedding_model_dims": 768,
                "index_method": "hnsw",
                "index_measure": "cosine_distance",
            },
        },
    }


def create_mem0_store(config: dict[str, Any]):
    """Initialize Mem0 lazily so unit tests and disabled memory need no dependency."""
    from mem0 import Memory

    return Memory.from_config(config)
