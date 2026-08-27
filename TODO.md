# NeuroAppAgentic — TODO

## 1. Expand the evaluation set from 10 to 100 questions

`eval/agentic_eval_10.jsonl` holds 10 questions (7 answerable + 3 refusals) as a starting
point. `eval/neuro_qa_100.jsonl` is already in this repo with all 100 grounded questions,
each carrying `source_file`, `page` and `gold_content_key`.

**Why it matters.** At n=10 the resolution is 0.1 — one flipped question moves any metric a
full step — and the judge was measured to vary **±0.1 on identical inputs** (two runs of an
identical configuration scored 0.70 and 0.60 on correctness). Almost every difference this
set can produce sits inside that noise. Comparing the agentic graph against the linear
baseline is exactly the kind of question n=10 cannot answer.

**What to do.** Build `eval/agentic_eval_100.jsonl` from all 100 answerable questions plus
roughly 15–20 out-of-corpus refusals, keeping the ratio near the current 70/30.

**Do these first, or the run becomes impractical:**
- Cost scales linearly and this graph already costs 4–9 model calls per question. 100
  questions could mean 600–900 calls; at local speeds that is overnight.
- Drop `retrieval_relevance` if it reads 1.00 across the board again — on a previous
  six-experiment run it discriminated nothing and cost ~90 minutes.
- Use `retrieval_recall` for retrieval tuning instead of the LLM judges: it is
  deterministic and runs in seconds, which is what makes `rrf_k`, `candidate_k` and chunk
  size tunable at all.

## 2. Measure whether the LLM router beats the heuristic

`src/router.py` computes `literal_score()` alongside the LLM's choice but does not use it.
Log both over a full run and compare. If the heuristic agrees with the LLM most of the
time, the router can be deterministic — removing one model call per question.

## 3. Measure the graph against the linear baseline

Nothing yet establishes that self-correction beats refusing early. Run
`eval/agentic_eval_10.jsonl` through both this graph and NeuroApp's linear pipeline, same
backend, same judge, and compare — particularly `abstention`, which the linear app scored
14/14 on its own set.

## 4. Restore Pinecone as the backend once quota resets

Chroma is a fallback with different chunking, so `gold_content_key` cannot be scored
against it and results are not comparable to any recorded number.
