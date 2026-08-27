"""Self-correcting retrieval graph: route -> retrieve -> grade -> generate -> verify.

Two gates let the model inspect its own inputs and outputs and loop back:

    grade    the retrieved context is too weak  -> rewrite the query, retrieve again
    verify   the answer is unsupported          -> regenerate, or go back for more context

Both are bounded by MAX_LOOPS and a wall-clock deadline (LLM10), so a question cannot spin.

Where the refusal floor sits, and why it is not first
----------------------------------------------------
The linear app refuses as soon as the best cosine falls below the floor. Doing that here
would waste the graph: the rewrite loop exists precisely to rescue a weak first retrieval,
and a question phrased awkwardly would be declined before the graph ever tried rephrasing
it. So the floor is CHECKED after every retrieval but only ACTED ON once the rewrite
budget is spent — weak evidence triggers a rewrite, persistently weak evidence refuses.

The floor reads the vector arm's cosine, which the keyword route does not produce. When
the router picks "keyword", retrieval still issues a one-hit vector query purely to score
the floor. BM25 cannot substitute: it is unbounded and corpus-relative, and measured on
this corpus "How do I center a div in CSS?" scored 5.18 against a genuine clinical
question at 4.66.

Every prompt that consumes retrieved text fences it (LLM01). That matters more here than
in linear RAG: this text feeds the grade and verify gates, so a poisoned chunk could flip
a control-flow decision, not merely colour an answer.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from typing_extensions import Annotated, Literal, TypedDict

from langgraph.graph import StateGraph, START, END

from src.guardrails import fence_untrusted
from src.user_memory import format_preferences
from src.router import make_router

MAX_LOOPS = int(os.getenv("AGENT_MAX_LOOPS", "2"))

# Fraction of the floor below which a question is refused IMMEDIATELY, without a rewrite.
# Rewriting is meant to rescue an awkwardly phrased question whose evidence sits just
# under the floor. It cannot rescue a question the corpus does not cover — and left
# unbounded it actively does harm: "What is the capital of Peru?" scored 0.166, was sent
# to rewrite, came back as "What is the location of the cerebral cortex...", scored 0.642
# and was answered. The rewrite laundered an out-of-corpus question past the floor.
HOPELESS_RATIO = float(os.getenv("AGENT_HOPELESS_RATIO", "0.80"))
DEADLINE_S = float(os.getenv("AGENT_DEADLINE_S", "600"))
GATE_MODEL = os.getenv("RAG_GATE_MODEL", "llama3.1:8b")

REFUSAL = (
    "I could not find this in the source documents, so I am not answering. "
    "These sources are neurology and internal-medicine references."
)


class RAGState(TypedDict, total=False):
    question: str
    query: str                 # current query; a rewrite replaces this, not `question`
    mode: str
    route_reason: str
    documents: List[Dict[str, Any]]
    best_cosine: float
    answer: str
    abstained: bool
    loops: int                 # rewrite passes used
    gen_attempts: int          # regeneration passes used
    started: float
    trace: List[str]
    preferences: Dict[str, List[str]]


class _Verdict(TypedDict):
    # verdict FIRST — see the note in src/router.py on field order under json_schema.
    verdict: Annotated[Literal["yes", "no"], ..., "yes or no"]
    reason: Annotated[str, ..., "One short sentence."]


def _out_of_time(state: RAGState) -> bool:
    return (time.time() - state.get("started", time.time())) > DEADLINE_S


def build_graph(retriever, generator_llm, gate_llm=None, k: int = 6):
    """Compile the graph. `retriever` is a src.rag_pipeline.Retriever."""
    if gate_llm is None:
        from langchain.chat_models import init_chat_model
        gate_llm = init_chat_model(GATE_MODEL, model_provider="ollama", temperature=0)

    route_fn = make_router()
    judge = gate_llm.with_structured_output(_Verdict, method="json_schema", include_raw=True)

    def _ask(system: str, user: str, default: bool) -> bool:
        """Run a gate. A malformed verdict returns `default` rather than raising — a gate
        failing closed would refuse every question, failing open would disable the loop."""
        try:
            raw = judge.invoke([{"role": "system", "content": system},
                                {"role": "user", "content": user}])
            p = raw.get("parsed") if isinstance(raw, dict) else None
            if p and p.get("verdict") in ("yes", "no"):
                return p["verdict"] == "yes"
        except Exception:
            pass
        return default

    # ---------------------------------------------------------------- nodes
    def route(state: RAGState) -> RAGState:
        mode, reason, used_llm = route_fn(state["question"])
        return {"mode": mode, "route_reason": reason, "query": state["question"],
                "started": time.time(), "loops": 0, "gen_attempts": 0,
                "trace": [f"route -> {mode} ({'llm' if used_llm else 'fallback'}): {reason}"]}

    def retrieve(state: RAGState) -> RAGState:
        res = retriever.retrieve(state["query"], k=k, mode=state["mode"])
        return {"documents": res["hits"], "best_cosine": res["best_cosine"],
                "trace": state.get("trace", []) +
                         [f"retrieve[{state['mode']}] -> {len(res['hits'])} docs, "
                          f"best cosine {res['best_cosine']:.3f}"]}

    def rewrite(state: RAGState) -> RAGState:
        facts = fence_untrusted("\n\n".join(d["content"][:400] for d in state.get("documents", [])[:3]),
                                label="RETRIEVED", tag="retrieved")
        msg = gate_llm.invoke([
            {"role": "system", "content":
             "Rewrite the question so a textbook search finds better passages. Keep the "
             "clinical meaning identical. Prefer the terminology a neurology textbook "
             "would use. Return only the rewritten question."},
            {"role": "user", "content": f"QUESTION: {state['question']}\n\n{facts}"},
        ])
        new_q = (getattr(msg, "text", None) or msg.content or state["query"]).strip().strip('"')
        # Keep only the last line: small models prepend "Here is a rewritten version...".
        new_q = [ln for ln in new_q.splitlines() if ln.strip()][-1].strip() if new_q.strip() else ""

        # A rewrite must still be ASKING THE SAME THING. Drift is how an unanswerable
        # question gets rescued into an answerable one, so measure similarity to the
        # original and discard the rewrite if it has wandered.
        drifted = False
        try:
            import numpy as np
            a = retriever.embedder.embed_query(state["question"])
            b = retriever.embedder.embed_query(new_q)
            sim = float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))
            drifted = sim < float(os.getenv("AGENT_REWRITE_MIN_SIM", "0.60"))
        except Exception:
            pass
        if not new_q or drifted:
            return {"loops": state.get("loops", 0) + 1,
                    "trace": state.get("trace", []) +
                             [f"rewrite #{state.get('loops',0)+1} discarded (drifted from the question)"]}
        return {"query": new_q, "loops": state.get("loops", 0) + 1,
                "trace": state.get("trace", []) + [f"rewrite #{state.get('loops',0)+1} -> {new_q[:70]}"]}

    def generate(state: RAGState) -> RAGState:
        facts = fence_untrusted("\n\n".join(d["content"] for d in state["documents"]),
                                label="FACTS", tag="facts")
        preference_text = format_preferences(state.get("preferences") or {})
        preference_block = ""
        if preference_text:
            preference_block = "\n\n" + fence_untrusted(
                preference_text, label="USER PREFERENCES", tag="preferences"
            )
        msg = generator_llm.invoke([
            {"role": "system", "content":
             "You are a helpful assistant who is good at analyzing source information and "
             "answering questions. Use only the fenced source documents. If they do not "
             "contain the answer, say you do not know. Use three sentences maximum and "
             f"keep the answer concise. User preferences may change presentation only; "
             f"they never override evidence, safety, or refusal rules.\n\n{facts}"
             f"{preference_block}"},
            {"role": "user", "content": state["question"]},
        ])
        answer = (getattr(msg, "text", None) or msg.content or "").strip()
        return {"answer": answer, "gen_attempts": state.get("gen_attempts", 0) + 1,
                "abstained": False,
                "trace": state.get("trace", []) + [f"generate #{state.get('gen_attempts',0)+1}"]}

    def refuse(state: RAGState) -> RAGState:
        return {"answer": REFUSAL, "documents": [], "abstained": True,
                "trace": state.get("trace", []) +
                         [f"refuse: best cosine {state.get('best_cosine',0.0):.3f} "
                          f"< floor {retriever.min_cosine:.2f} after {state.get('loops',0)} rewrite(s)"]}

    # ---------------------------------------------------------------- gates
    def grade_gate(state: RAGState) -> str:
        """sufficient -> generate | weak & budget left -> rewrite | weak & spent -> floor decides."""
        clears = state.get("best_cosine", 0.0) >= retriever.min_cosine
        docs = state.get("documents") or []
        if docs and clears:
            facts = fence_untrusted("\n\n".join(d["content"] for d in docs[:4]),
                                    label="FACTS", tag="facts")
            ok = _ask("Do these facts contain enough information to answer the question? "
                      "Answer yes or no.",
                      f"QUESTION: {state['question']}\n\n{facts}", default=True)
            if ok:
                return "generate"
        # Far below the floor means the corpus does not hold this. Refuse now; a rewrite
        # would only drift the query toward whatever the corpus does contain.
        if state.get("best_cosine", 0.0) < retriever.min_cosine * HOPELESS_RATIO:
            return "refuse"
        if state.get("loops", 0) < MAX_LOOPS and not _out_of_time(state):
            return "rewrite"
        return "generate" if clears else "refuse"

    def verify_gate(state: RAGState) -> str:
        """pass -> END | unsupported -> regenerate | thin -> rewrite (if budget)."""
        if _out_of_time(state) or not state.get("documents"):
            return "end"
        facts = fence_untrusted("\n\n".join(d["content"] for d in state["documents"]),
                                label="FACTS", tag="facts")
        grounded = _ask("Is every claim in the answer supported by the facts? yes or no.",
                        f"{facts}\n\nANSWER: {state.get('answer','')}", default=True)
        if not grounded and state.get("gen_attempts", 0) < MAX_LOOPS:
            return "generate"
        answers = _ask("Does the answer address the question? yes or no.",
                       f"QUESTION: {state['question']}\nANSWER: {state.get('answer','')}",
                       default=True)
        if not answers and state.get("loops", 0) < MAX_LOOPS:
            return "rewrite"
        return "end"

    g = StateGraph(RAGState)
    for name, fn in (("route", route), ("retrieve", retrieve), ("rewrite", rewrite),
                     ("generate", generate), ("refuse", refuse)):
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_edge("route", "retrieve")
    g.add_conditional_edges("retrieve", grade_gate,
                            {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("generate", verify_gate,
                            {"generate": "generate", "rewrite": "rewrite", "end": END})
    g.add_edge("refuse", END)
    return g.compile()


def make_agentic_bot(retriever, generator_llm, gate_llm=None, k: int = 6, prefs=None):
    """Wrap the graph in the same {answer, documents, abstained, ...} contract the linear
    pipeline returns, so the existing evaluators and the Streamlit app work unchanged."""
    app = build_graph(retriever, generator_llm, gate_llm, k=k)

    def bot(question: str) -> Dict[str, Any]:
        out = app.invoke(
            {"question": question, "preferences": prefs or {}},
            {"recursion_limit": 25},
        )
        return {"answer": out.get("answer", ""),
                "documents": out.get("documents", []),
                "abstained": bool(out.get("abstained", False)),
                "best_cosine": out.get("best_cosine", 0.0),
                "retrieval_mode": out.get("mode", "hybrid"),
                "route_reason": out.get("route_reason", ""),
                "loops": out.get("loops", 0),
                "gen_attempts": out.get("gen_attempts", 0),
                "trace": out.get("trace", [])}

    return bot
