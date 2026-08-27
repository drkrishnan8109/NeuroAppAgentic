"""Retrieval-mode router: an LLM picks vector, keyword or hybrid per question.

Why route at all
----------------
No single retrieval mode wins on this corpus. Measured on a 10-question sweep with
generation and judging held constant:

    keyword  0.90 correctness
    vector   0.70
    hybrid   0.60

The reason is query-dependent, which is exactly what makes it routable. `tofersen SOD1`
scored a vector cosine of 0.284 — the embedding had essentially nothing — while BM25
returned confident literal matches. A paraphrased conceptual question inverts that. And
RRF hybrid came last because fusion scores by RANK: a chunk at ranks 3 and 4 in both arms
(1/63 + 1/64 = 0.0315) outranks the right answer at rank 1 in one arm (1/61 = 0.0164), so
agreement beats precision and the stronger arm gets diluted.

Routing per question is an attempt to get keyword's precision on literal queries and
vector's reach on conceptual ones, instead of averaging them.

Local model by design: this runs once per question on top of the graph's other calls, so
it uses the smallest model available (llama3.1:8b) rather than the generator.
"""

from __future__ import annotations

import os
import re
from typing import Literal, Optional

from typing_extensions import Annotated, TypedDict

Mode = Literal["vector", "keyword", "hybrid"]

ROUTER_MODEL = os.getenv("RAG_ROUTER_MODEL", "llama3.1:8b")

# Cheap pre-check. Rare literal tokens are the clearest keyword signal, and spotting them
# needs no model: drug names, gene symbols and eponymous criteria look distinctive.
# Used only to record a second opinion alongside the LLM's, so the two can be compared —
# the LLM decides.
_LITERAL_HINT = re.compile(
    r"\b("
    r"[A-Z]{2,}\d*|"                 # SOD1, EEG, CIDP, NMDA
    r"\w+(?:mab|nib|zumab|ximab)\b|" # monoclonals / inhibitors
    r"\w+(?:sen|parin|olol|azepam|pridone|dopa)\b"   # tofersen, heparin, levodopa
    r")\b"
)


def literal_score(question: str) -> int:
    """How many rare-literal tokens the question carries. Diagnostic only."""
    return len(set(_LITERAL_HINT.findall(question)))


class RouteChoice(TypedDict):
    # mode FIRST, deliberately. json_schema constrained decoding fills fields in schema
    # order, and a small model asked to explain first will commit its answer into the
    # explanation and then emit something arbitrary for the field that matters. Observed
    # exactly that: reason="keyword" while mode came out "vector".
    mode: Annotated[Mode, ..., "vector, keyword or hybrid"]
    reason: Annotated[str, ..., "One short sentence explaining the choice."]


ROUTER_INSTRUCTIONS = """You choose how to search a neurology textbook corpus for a question.

Pick exactly one:

- "keyword" when the question contains rare literal terms that must match exactly: drug
  names, gene symbols, eponymous criteria, scale names, specific test names. Exact string
  matching finds these; embeddings blur them into neighbours.
- "vector" when the question is conceptual or paraphrased and carries no distinctive
  literal term — asking about a mechanism, a general approach, or a comparison.
- "hybrid" only when the question genuinely needs both, or you cannot tell.

Answer with the mode and one short sentence of reasoning."""


def make_router(llm=None):
    """Return route(question) -> (mode, reason, used_llm).

    Falls back to a deterministic choice when the model is unreachable or returns
    something unparseable, so a router failure degrades to hybrid rather than taking the
    graph down. hybrid is the safe default: it is the only mode that consults both arms.
    """
    if llm is None:
        from langchain.chat_models import init_chat_model
        llm = init_chat_model(ROUTER_MODEL, model_provider="ollama", temperature=0)

    grader = llm.with_structured_output(RouteChoice, method="json_schema", include_raw=True)

    def route(question: str):
        try:
            raw = grader.invoke([
                {"role": "system", "content": ROUTER_INSTRUCTIONS},
                {"role": "user", "content": f"QUESTION: {question}"},
            ])
            parsed = raw.get("parsed") if isinstance(raw, dict) else None
            if parsed and parsed.get("mode") in ("vector", "keyword", "hybrid"):
                return parsed["mode"], (parsed.get("reason") or "").strip(), True
        except Exception as exc:
            return "hybrid", f"router unavailable ({type(exc).__name__}); defaulting to hybrid", False
        return "hybrid", "router returned no usable mode; defaulting to hybrid", False

    return route
