"""Offline evaluation: score to local disk, with LangSmith as an optional later upload.

The LangSmith-driven harness lost a 10-question run: DNS failures mid-judging stalled the
loop for 9 hours and only 1 of 10 rows was ever persisted. Measurement should not depend on
a network round trip per row, so this writes every answer and every score to JSONL as it
goes. A crash or an outage costs the current row, not the run.

Two phases, as before, so Ollama loads each model once rather than swapping per question.
"""
import json, os, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_root, "..", "NeuroApp", ".env"))
os.environ["LANGSMITH_TRACING"] = "false"      # no per-call network traffic

from langchain.chat_models import init_chat_model
from src.rag_pipeline import Retriever, get_backend, build_bm25_local
from src.agent_graph import make_agentic_bot

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.getenv("AGENTIC_EVAL_FILE", "agentic_eval_100.jsonl"))
GEN_OUT = os.path.join(HERE, "runs", "generations.jsonl")
SCORE_OUT = os.path.join(HERE, "runs", "scores.jsonl")
os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)

rows = [json.loads(l) for l in open(DATA)]
print(f"loaded {len(rows)} questions from {os.path.basename(DATA)}", flush=True)

# ------------------------------------------------------------------ phase 1
done = set()
if os.path.exists(GEN_OUT):                      # resume rather than redo
    for l in open(GEN_OUT):
        done.add(json.loads(l)["question"])
    print(f"resuming: {len(done)} already generated", flush=True)

backend = get_backend()
print(f"backend={backend.name} floor={backend.min_cosine} vectors={backend.store.count()}", flush=True)
retriever = Retriever.from_backend(backend, build_bm25_local(backend.chunk_size, backend.chunk_overlap))
bot = make_agentic_bot(retriever, init_chat_model("qwen2.5:14b", model_provider="ollama", temperature=0.0))

print("\n=== PHASE 1 — generating ===", flush=True)
t0 = time.time()
with open(GEN_OUT, "a") as fh:
    for i, r in enumerate(rows, 1):
        q = r["inputs"]["question"]
        if q in done:
            continue
        try:
            out = bot(q)
        except Exception as exc:
            out = {"answer": "", "documents": [], "abstained": False, "error": f"{type(exc).__name__}: {exc}"}
        fh.write(json.dumps({
            "question": q,
            "reference": r["outputs"]["answer"],
            "answerable": r["outputs"]["answerable"],
            "gold_source": (r.get("metadata") or {}).get("source_file"),
            "gold_page": (r.get("metadata") or {}).get("page"),
            "answer": out.get("answer", ""),
            "abstained": bool(out.get("abstained", False)),
            "mode": out.get("retrieval_mode"),
            "route_reason": out.get("route_reason", ""),
            "loops": out.get("loops", 0),
            "gen_attempts": out.get("gen_attempts", 0),
            "best_cosine": out.get("best_cosine", 0.0),
            "docs": [{"content": d.get("content", ""),
                      "source_file": (d.get("metadata") or {}).get("source_file"),
                      "page": (d.get("metadata") or {}).get("page")}
                     for d in (out.get("documents") or [])],
            "error": out.get("error"),
        }) + "\n")
        fh.flush()
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)
print(f"PHASE1_DONE in {time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------ phase 2
try:
    urllib.request.urlopen(urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": "qwen2.5:14b", "keep_alive": 0}).encode(),
        headers={"Content-Type": "application/json"}), timeout=60).read()
except Exception as exc:
    print(f"  could not unload generator ({exc})", flush=True)

from src.evaluators import correctness, groundedness, relevance, retrieval_relevance, abstention

gens = [json.loads(l) for l in open(GEN_OUT)]
scored = set()
if os.path.exists(SCORE_OUT):
    for l in open(SCORE_OUT):
        scored.add(json.loads(l)["question"])
    print(f"resuming: {len(scored)} already scored", flush=True)

print(f"\n=== PHASE 2 — judging {len(gens)-len(scored)} rows ===", flush=True)
t0 = time.time()
with open(SCORE_OUT, "a") as fh:
    for i, g in enumerate(gens, 1):
        if g["question"] in scored:
            continue
        inputs = {"question": g["question"]}
        outputs = {"answer": g["answer"], "documents": g["docs"], "abstained": g["abstained"]}
        reference = {"answer": g["reference"], "answerable": g["answerable"]}
        s = {"question": g["question"], "answerable": g["answerable"],
             "mode": g["mode"], "loops": g["loops"], "gen_attempts": g["gen_attempts"],
             # deterministic, free: did retrieval surface the page the answer came from?
             # Page-level rather than chunk-hash, so it works across chunkings.
             "page_recall": (None if not g["answerable"] else
                             float(any(d.get("source_file") == g["gold_source"]
                                       and d.get("page") == g["gold_page"] for d in g["docs"]))),
             "abstention": float(abstention(inputs, outputs, reference))}
        for name, fn in (("correctness", lambda: correctness(inputs, outputs, reference)),
                         ("groundedness", lambda: groundedness(inputs, outputs, reference)),
                         ("relevance", lambda: relevance(inputs, outputs, reference)),
                         ("retrieval_relevance", lambda: retrieval_relevance(inputs, outputs, reference))):
            try:
                s[name] = float(fn())
            except Exception as exc:
                s[name] = None
                s[name + "_error"] = type(exc).__name__
        fh.write(json.dumps(s) + "\n"); fh.flush()
        if i % 10 == 0:
            print(f"  {i}/{len(gens)}  ({time.time()-t0:.0f}s)", flush=True)
print(f"PHASE2_DONE in {time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
