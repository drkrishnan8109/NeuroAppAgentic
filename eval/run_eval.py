"""Two-phase evaluation of the agentic graph.

Phase 1 generates every answer with the graph (generator + gate models resident), phase 2
judges the stored runs. Splitting them means Ollama loads two models once each instead of
swapping between them per question — an interleaved run of this shape previously took
15.5 hours and produced zero scores.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "NeuroApp", ".env"))

from langchain.chat_models import init_chat_model
from langsmith import Client
from langsmith.evaluation import evaluate_existing

from src.rag_pipeline import Retriever, get_backend, build_bm25_local
from src.agent_graph import make_agentic_bot
from src.evaluators import evaluators

DATASET = os.getenv("AGENTIC_DATASET", "NeuroAppAgentic Eval v1")
HERE = os.path.dirname(os.path.abspath(__file__))

client = Client()
if not client.has_dataset(dataset_name=DATASET):
    rows = [json.loads(l) for l in open(os.path.join(HERE, "agentic_eval_10.jsonl"))]
    ds = client.create_dataset(dataset_name=DATASET)
    client.create_examples(dataset_id=ds.id,
                           inputs=[r["inputs"] for r in rows],
                           outputs=[r["outputs"] for r in rows])
    print(f"created dataset '{DATASET}' with {len(rows)} examples", flush=True)
else:
    print(f"reusing dataset '{DATASET}' "
          f"({client.read_dataset(dataset_name=DATASET).example_count} examples)", flush=True)

backend = get_backend()
print(f"backend={backend.name} floor={backend.min_cosine} vectors={backend.store.count()}", flush=True)
retriever = Retriever.from_backend(backend, build_bm25_local(backend.chunk_size, backend.chunk_overlap))
bot = make_agentic_bot(
    retriever,
    init_chat_model("qwen2.5:14b", model_provider="ollama", temperature=0.0),
    prefs=None,
)


def loop_count(outputs: dict) -> dict:
    """Deterministic, no model: how much self-correction each question needed."""
    return {"key": "loop_count",
            "score": float(outputs.get("loops", 0) + outputs.get("gen_attempts", 0))}


print("\n=== PHASE 1 — generating (no judging) ===", flush=True)
t = time.time()
res = client.evaluate(lambda inputs: bot(inputs["question"]),
                      data=DATASET, experiment_prefix="agentic-graph",
                      max_concurrency=1,
                      metadata={"pipeline": "langgraph route/grade/rewrite/verify",
                                "generation": "qwen2.5:14b", "gates": "llama3.1:8b",
                                "backend": backend.name, "floor": backend.min_cosine})
df = res.to_pandas()
ok = int(df["outputs.answer"].notna().sum()) if "outputs.answer" in df.columns else 0
print(f"  {res.experiment_name}: {ok}/{len(df)} answers in {time.time()-t:.0f}s", flush=True)
if ok != len(df):
    raise SystemExit("phase 1 incomplete — judging would produce nan scores")

print("\n=== PHASE 2 — judging ===", flush=True)
import urllib.request
try:  # free the generator before the judge loads
    urllib.request.urlopen(urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": "qwen2.5:14b", "keep_alive": 0}).encode(),
        headers={"Content-Type": "application/json"}), timeout=60).read()
except Exception as e:
    print(f"  could not unload generator ({e}) — continuing", flush=True)

t = time.time()
judged = evaluate_existing(res.experiment_name, evaluators=evaluators + [loop_count],
                           client=client, max_concurrency=0)
print(f"  judged in {time.time()-t:.0f}s", flush=True)
print(f"\nDONE experiment={res.experiment_name}", flush=True)
