"""Clinician-facing front end for the agentic neurology RAG.

Same contract as NeuroApp's app — question in, cited answer or a refusal out, everything
wrapped in guard() — with one addition: the graph makes decisions, so the UI shows them.
A reader can see which retrieval mode the router chose and why, whether the query was
rewritten, and whether the answer was regenerated. Without that, a self-correcting system
is a black box that occasionally takes ten times longer for no visible reason.
"""

import os
import sys
import threading
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
_here = Path(__file__).resolve().parent.parent
load_dotenv(_here / ".env")
load_dotenv(_here.parent / "NeuroApp" / ".env")     # shared local keys, if present

# Community Cloud has no .env: secrets arrive as st.secrets. Copy them into the
# environment so every os.getenv() in src/ keeps working unchanged.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

st.set_page_config(page_title="Neurology Assistant — Agentic", page_icon="🧠", layout="centered")


def _login_screen():
    st.title("Neurology Assistant")
    st.caption("Sign in to keep your presentation preferences private and available across sessions.")
    st.button("Log in with Google", on_click=st.login, args=("google",), type="primary")


if not getattr(st.user, "is_logged_in", False):
    _login_screen()
    st.stop()


# Import the model and retrieval stack only after authentication. This avoids starting
# indexes and model clients for unauthenticated visitors.
from langchain.chat_models import init_chat_model

from src.agent_graph import MAX_LOOPS, make_agentic_bot
from src.guardrails import GuardrailConfig, guard
from src.memory_runtime import build_mem0_config, create_mem0_store, make_preference_extractor
from src.memory_schema import build_user_id, should_extract_preferences
from src.rag_pipeline import Retriever, build_bm25_local, get_backend
from src.user_memory import UserMemory, build_subject_hash

GENERATORS = [g.strip() for g in os.getenv(
    "RAG_GENERATORS",
    "ollama:qwen2.5:14b,mistralai:mistral-small-latest,google_genai:gemini-flash-latest",
).split(",") if g.strip()]


@st.cache_resource(show_spinner="Connecting to the index…")
def _backend():
    return get_backend()


@st.cache_resource(show_spinner="Building the keyword index…")
def _bm25(chunk_size: int, chunk_overlap: int):
    # Always built: the router can choose keyword or hybrid on any question, so the arm
    # has to be there. From the local PDF cache, which takes seconds and costs no egress.
    return build_bm25_local(chunk_size, chunk_overlap)


@st.cache_resource(show_spinner="Starting the model…")
def _llm(model_id: str):
    return init_chat_model(model_id, temperature=0.0)


@st.cache_resource(show_spinner="Connecting to preference memory…")
def _memory_store(connection_string: str, google_api_key: str):
    # The store is shared infrastructure; user identity and preferences are never cached.
    return create_mem0_store(build_mem0_config(connection_string, google_api_key))


def _user_memory():
    connection_string = os.getenv("SUPABASE_DB_URL", "")
    google_api_key = os.getenv("GOOGLE_API_KEY", "")
    if not connection_string or not google_api_key:
        return None
    extractor_model = os.getenv(
        "MEMORY_EXTRACTOR_MODEL", "google_genai:gemini-2.5-flash-lite"
    )
    return UserMemory(
        _memory_store(connection_string, google_api_key),
        extractor=make_preference_extractor(_llm(extractor_model)),
    )


def _identity():
    subject = str(st.user.get("sub", ""))
    pepper = os.getenv("MEMORY_USER_PEPPER", "")
    return build_user_id(subject, pepper), build_subject_hash(subject)


try:
    user_id, subject_hash = _identity()
    memory = _user_memory()
    memory_error = "" if memory else "SUPABASE_DB_URL or GOOGLE_API_KEY is not configured"
except Exception as exc:
    user_id, subject_hash, memory = "", "", None
    memory_error = f"memory initialization failed ({type(exc).__name__})"

if st.session_state.get("memory_user_id") != user_id:
    st.session_state["memory_user_id"] = user_id
    st.session_state["preferences"] = (
        memory.load_preferences(user_id) if memory and user_id else {}
    )


def _answer(retriever, question, k, preferences=None):
    """Try each generator in turn; the gates always run on the small local model."""
    errors = []
    for model_id in GENERATORS:
        try:
            bot = make_agentic_bot(
                retriever, _llm(model_id), k=k, prefs=preferences or {}
            )
            return guard(bot, GuardrailConfig())(question), model_id, errors
        except Exception as exc:
            errors.append(f"{model_id}: {type(exc).__name__} {str(exc)[:110]}")
    raise RuntimeError("All generators failed:\n" + "\n".join(errors))


with st.sidebar:
    st.caption(f"Signed in as **{st.user.get('name', 'user')}**")
    st.button("Log out", on_click=st.logout)
    st.divider()
    st.subheader("Remembered preferences")
    preferences = st.session_state.get("preferences", {})
    if memory_error:
        st.caption(f"Memory unavailable: {memory_error}")
    elif not preferences:
        st.caption("No preferences remembered yet.")
    else:
        for category, facts in preferences.items():
            st.caption(f"**{category.replace('_', ' ').title()}**")
            for fact in facts:
                st.write(f"· {fact}")
        confirm_delete = st.checkbox("Confirm deleting all my preferences")
        if st.button("Delete my preferences", disabled=not confirm_delete):
            try:
                memory.delete_all(user_id)
                st.session_state["preferences"] = {}
                st.success("Preferences deleted.")
                st.rerun()
            except Exception:
                st.error("Preferences could not be deleted right now.")
    st.divider()
    st.subheader("Retrieval")
    st.caption(
        "The retrieval mode is **chosen per question by an agent** — it is not a setting. "
        "Keyword for rare literal terms (drug names, gene symbols), vector for conceptual "
        "questions, hybrid when both or unsure."
    )
    k = st.slider("Passages to use", 3, 10, 6,
                  help="How many retrieved chunks are placed in the model's prompt.")
    st.divider()
    b = _backend()
    st.caption(f"**Index:** `{b.name}` — {b.store.count():,} vectors, {b.chunk_size}/{b.chunk_overlap} chunks")
    st.caption(
        f"**Answer floor:** cosine ≥ {b.min_cosine:.2f}. Questions whose best passage falls "
        f"well below it are declined immediately; those just under it get up to {MAX_LOOPS} "
        "rewrite attempts first."
    )
    st.caption("**Gates:** `llama3.1:8b` · **Generation:** first available of "
               + ", ".join(f"`{g}`" for g in GENERATORS))

st.title("Neurology Assistant")
st.caption(
    "Answers come only from four indexed references — Bradley's *Neurology in Clinical "
    "Practice*, *Clinical Neurophysiology*, Harrison's *Neurology in Clinical Medicine*, "
    "and the *Handbook of Emergency Neurology*. The assistant re-searches when the "
    "evidence is thin, and declines when the corpus does not cover the question."
)

question = st.text_area("Clinical question", height=90,
                        placeholder="e.g. What is the management of GBS if there is no improvement after IVIG?")
if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        b = _backend()
        retriever = Retriever.from_backend(b, _bm25(b.chunk_size, b.chunk_overlap))
        with st.spinner("Searching, checking, and re-searching if needed…"):
            result, used_model, tried = _answer(
                retriever, question, k, st.session_state.get("preferences", {})
            )
    except Exception as exc:
        st.error(f"Could not answer that: {exc}")
        st.stop()

    if (result.get("guardrails") or {}).get("blocked"):
        st.warning("That request was blocked before reaching the model.")
        st.write(result["answer"]); st.stop()

    if result.get("abstained"):
        st.info("**Not covered by the indexed references.**")
        st.write(result["answer"])
        st.caption(f"Best passage scored cosine {result.get('best_cosine', 0):.3f}, below the "
                   f"{_backend().min_cosine:.2f} answer floor.")
    else:
        st.markdown(result["answer"])

    # --- what the agent decided -------------------------------------------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Route", result.get("retrieval_mode", "—"),
              help="Chosen by the router from the question's wording.")
    c2.metric("Rewrites", result.get("loops", 0),
              help=f"Times the query was reformulated after weak retrieval (max {MAX_LOOPS}).")
    c3.metric("Generations", result.get("gen_attempts", 0),
              help=f"Times the answer was written; more than one means the verify gate "
                   f"rejected an unsupported draft (max {MAX_LOOPS}).")
    if result.get("route_reason"):
        st.caption(f"**Why this route:** {result['route_reason']}")

    with st.expander("Decision trace"):
        for step in result.get("trace", []):
            st.write(f"· {step}")
        st.caption(f"Answered by `{used_model}`"
                   + (f" after {len(tried)} generator(s) failed" if tried else ""))

    docs = result.get("documents") or []
    if docs:
        st.divider(); st.subheader("Sources")
        for i, d in enumerate(docs, 1):
            m = d.get("metadata") or {}
            bits = []
            if d.get("similarity") is not None: bits.append(f"cosine {d['similarity']:.3f}")
            if d.get("bm25_score") is not None: bits.append(f"BM25 {d['bm25_score']:.2f}")
            if d.get("fusion_score") is not None: bits.append(f"RRF {d['fusion_score']:.4f}")
            with st.expander(f"{i}. {m.get('source_file','unknown')} · p.{m.get('page','?')}"
                             f"  —  {' · '.join(bits)}"):
                st.write(d["content"])

    # A daemon write keeps extraction and Mem0 off the answer latency path. Only turns
    # with explicit first-person preference language reach the extractor.
    if memory and user_id and should_extract_preferences(question):
        threading.Thread(
            target=memory.remember_from_text,
            args=(question, user_id, subject_hash),
            daemon=True,
            name="preference-memory-write",
        ).start()

st.divider()
st.caption("⚕️ Decision support only — not a diagnosis, and no substitute for clinical "
           "judgement or current local guidelines. Verify against the cited page before acting.")
