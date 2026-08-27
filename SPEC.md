# NeuroAppAgentic — specification

A conversion of NeuroApp's linear retrieve-then-generate pipeline into a self-correcting
agentic one. Nothing in `NeuroApp` or `drkrag` is modified; shared modules are copied, not
imported across repositories.

## 1. Router — an agent picks the retrieval mode

An LLM chooses `vector`, `keyword` or `hybrid` for each question before retrieval runs.

This is the central change, and it is motivated by measurement rather than fashion. On a
10-question sweep with generation and judging held constant:

| mode | correctness |
| --- | --- |
| keyword | **0.90** |
| vector | 0.70 |
| hybrid | 0.60 |

No mode wins universally, and the reason is query-dependent. `tofersen SOD1` scored a
vector cosine of **0.284** — the embedding found essentially nothing — while BM25 returned
confident literal matches. Conceptual, paraphrased questions invert that.

Hybrid came **last**, which is the counterintuitive part. RRF fuses by rank, so a chunk at
ranks 3 and 4 in both arms scores `1/63 + 1/64 = 0.0315` and outranks the correct passage
sitting at rank 1 in one arm at `1/61 = 0.0164`. Agreement beats precision by design, and
on this corpus that dilutes the stronger arm. Routing aims to get keyword's precision on
literal queries and vector's reach on conceptual ones instead of averaging them.

Runs on a local `llama3.1:8b` rather than the generator: it fires once per question on top
of the graph's other calls, so it uses the smallest capable model. A deterministic
`literal_score()` heuristic is computed alongside and logged, so the LLM's added value can
be measured later. Router failure degrades to `hybrid`, never to an exception.

## 2. Control flow (LangGraph)

```
route → retrieve → grade ──sufficient──→ generate → verify ──pass──→ answer
           ↑          │                     ↑          │
           └─ rewrite ┘ (≤2)                └──────────┘ (≤2 regenerate)
                      │                                 │
                      └──── floor fails, budget spent ──┴──→ refuse
```

`MAX_LOOPS = 2` caps rewrites and regenerations independently, on top of a wall-clock
deadline and a `recursion_limit` of 25 (LLM10: unbounded consumption).

Both gates fail **open** on a malformed verdict. A gate that failed closed would refuse
every question; one that failed open in the other direction would disable the loop.

## 3. The refusal floor is checked early and acted on late

The linear app refuses as soon as the best cosine falls below the floor. Doing that inside
a graph would waste the graph — the rewrite loop exists precisely to rescue weak
retrieval, and an awkwardly phrased question would be declined before the graph tried
rephrasing it.

So: the floor is evaluated after **every** retrieval, but only triggers a refusal once the
rewrite budget is spent. Weak evidence rewrites; persistently weak evidence refuses.

Floor values are per backend, because each is calibrated against its own embedding model's
score distribution and none of them transfer:

| backend | embedder | chunking | floor |
| --- | --- | --- | --- |
| Pinecone | BGE-M3, 1024-d | 1500/300 | 0.53 |
| Chroma | all-MiniLM-L6-v2, 384-d | 1000/200 | 0.43 |

The floor reads the **vector** arm, which the keyword route does not produce, so keyword
retrieval still issues a one-hit vector query purely to score it. BM25 cannot substitute:
it is unbounded and corpus-relative, and on this corpus "How do I center a div in CSS?"
scored 5.18 against a genuine clinical question at 4.66.

## 4. Guardrails

- **`fence_untrusted()` in every prompt that consumes retrieved text.** This matters more
  here than in linear RAG: the text feeds the grade and verify gates, so a poisoned chunk
  could flip a control-flow decision rather than merely colour an answer.
- **`guard()` on the serving path only** — never inside `client.evaluate`, because it
  rewrites text (redaction, disclaimer, citations) and would skew the judges.
- **`screen_documents_for_ingest` is deliberately NOT used.** Measured on this corpus it
  produced 0 true positives in 71 flags, dropping legitimate content including dantrolene
  dosing.

## 5. Models

| role | model | why |
| --- | --- | --- |
| router | `llama3.1:8b` (local) | one call per question; smallest capable |
| gates (grade, verify, rewrite) | `llama3.1:8b` (local) | 2–5 calls per question |
| generation | `qwen2.5:14b` (local) | the model every recorded score was produced with |

Splitting gates from generation is a direct response to cost: a question now costs 4–9
model calls instead of 1.

## 6. Evaluation

`eval/agentic_eval_10.jsonl` — 10 questions: **7 answerable** sampled from the
100-question grounded set (each carrying `gold_content_key`) and **3 out-of-corpus
refusals**. Testing a cosine floor with no refusal cases would leave the headline feature
unmeasured.

Metrics: the five from NeuroApp (`correctness`, `groundedness`, `relevance`,
`retrieval_relevance`, `abstention`) plus two deterministic, judge-free additions:

- **`retrieval_recall`** — did the gold chunk appear in the retrieved set?
- **`loop_count`** — how often the graph rewrote or regenerated.

## 7. Known constraints

- **Cost.** 4–9 model calls per question against 1 for the linear app. A 10-question eval
  may take 2–4 hours locally. Use the two-phase pattern: generate everything, unload, then
  judge.
- **Pinecone egress is exhausted for the month**, so this currently runs on Chroma at
  floor 0.43.
- **`gold_content_key` cannot be scored on Chroma.** The labels hash 1500/300 chunks;
  Chroma stores 1000/200, so every key misses by construction. Recall is measurable
  against Pinecone or against BM25 built at 1500/300.
