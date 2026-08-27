"""Serving-side RAG pipeline: retrieval, fusion, abstention, generation.

Why this module exists
----------------------
All of this previously lived inside a notebook cell, which meant nothing outside a
Jupyter kernel could answer a question — no app, no API, no test. The logic is lifted
here verbatim in behaviour so the notebooks and the Streamlit app share one
implementation and cannot drift apart.

Two things are deliberately pluggable, because the app must run in two very different
places:

  * THE EMBEDDER. Locally, sentence-transformers loads BAAI/bge-m3 (2.3 GB) onto Apple
    MPS. On Streamlit Community Cloud that will not fit in the container, so queries go
    through the HuggingFace Inference API instead — the SAME model, so the 1024-d
    vectors still match the index. Using a different embedding model would silently
    produce meaningless similarities against vectors built by bge-m3.
  * THE GENERATOR. Ollama is unreachable from a hosted container, so the model id is
    read from the environment and any LangChain-supported provider can be named.

The BM25 arm needs the chunk TEXT in memory, which the deployed app does not ship (the
corpus is licensed textbooks). `load_chunks_from_pinecone` rebuilds it from the text
already stored in the index metadata, so keyword retrieval survives deployment without
redistributing the source PDFs.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from src.pinecone_store import PineconeVectorStore, MAX_TOP_K, TEXT_KEY
from src.keyword_search import BM25Index, content_key

# Abstention floor. Calibrated on the bge-m3 / 1500-300 corpus:
#   cosine: in-corpus min 0.573 | out-of-corpus max 0.493 -> SEPARABLE, midpoint ~0.53
#   bm25:   in-corpus min 4.66  | out-of-corpus max 5.18  -> OVERLAPPING, unusable
# BM25 is excluded from the gate for the reason spelled out in `clears_floor`.
MIN_COSINE = float(os.getenv("RAG_MIN_COSINE", "0.53"))

DEFAULT_INDEX = os.getenv("PINECONE_INDEX", "drkrag-pdf-bge-m3")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = int(os.getenv("RAG_EMBED_DIM", "1024"))

_HF_URL = "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"



# ------------------------------------------------------------------ transient retries
MAX_NETWORK_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "5"))

# Errors worth retrying: the name did not resolve, the socket dropped, the request timed
# out. These are transient and usually clear within seconds.
_TRANSIENT = ("nameresolutionerror", "failed to resolve", "connection error",
              "connectionerror", "connection reset", "connection aborted",
              "temporary failure in name resolution", "timed out", "timeout",
              "max retries exceeded", "bad gateway", "service unavailable", "502", "503")

# Errors NOT worth retrying, however transient they look. A monthly quota does not clear
# on the next attempt, and retrying it just turns a clear failure into a slow one — which
# is exactly what happened when DNS failures stalled a judging loop for 9 hours.
_PERMANENT = ("egress limit", "quota", "resource_exhausted", "unauthorized", "forbidden",
              "invalid api key", "401", "403")


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    if any(p in text for p in _PERMANENT):
        return False
    return any(t in text for t in _TRANSIENT)


def with_retries(fn, *args, what: str = "request", attempts: int = None, **kwargs):
    """Call fn, retrying transient network failures up to `attempts` times, then give up.

    Bounded deliberately. An unbounded retry converts an outage into an indefinite hang:
    a run that should have failed in seconds instead ran for nine hours and recorded one
    row of ten. Failing after a fixed number of tries makes the outage visible.
    """
    attempts = attempts or MAX_NETWORK_RETRIES
    last = None
    for i in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            if not _is_transient(exc):
                raise
            if i == attempts:
                break
            delay = min(2 ** (i - 1), 8)
            print(f"  {what}: {type(exc).__name__} (attempt {i}/{attempts}), retrying in {delay}s")
            time.sleep(delay)
    raise RuntimeError(
        f"{what} failed after {attempts} attempts; last error: {type(last).__name__}: {last}"
    ) from last

# --------------------------------------------------------------------------- embedders
_HF_TOKEN_NAMES = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_KEY",
                   "HUGGINGFACE_API_KEY")


def hf_token() -> Optional[str]:
    """First HuggingFace token found under any of the names in common use.

    Kept in one place so the constructor and get_embedder() cannot disagree: gating the
    selector on HF_TOKEN while the constructor accepted HUGGING_FACE_KEY meant a valid
    token was present, the API embedder worked, and the local 2.3 GB model was loaded
    anyway — silently, with nothing to indicate the hosted path had been skipped.
    """
    return next((os.getenv(n) for n in _HF_TOKEN_NAMES if os.getenv(n)), None)


class HFInferenceEmbedder:
    """Embed queries through the HuggingFace Inference API.

    For a hosted deployment this replaces a 2.3 GB local model with an HTTP call. It must
    name the SAME model that built the index — a 1024-d vector from a different model is
    still 1024-d and will still return neighbours, just meaningless ones, which is the
    kind of failure that looks like poor retrieval rather than a bug.
    """

    def __init__(self, model: str = EMBED_MODEL, token: Optional[str] = None, timeout: int = 30):
        self.model = model
        self.token = token or hf_token()
        self.timeout = timeout

    def embed_query(self, text: str) -> np.ndarray:
        return with_retries(self._embed_once, text, what="HF embed")

    def _embed_once(self, text: str) -> np.ndarray:
        r = requests.post(
            _HF_URL.format(model=self.model),
            headers={"Authorization": f"Bearer {self.token}"},
            json={"inputs": text},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        # The endpoint returns either [dim] or [[dim]] depending on batching.
        vec = np.array(data[0] if isinstance(data[0], list) else data, dtype=np.float32)
        # The index holds normalised vectors, so normalise here too or cosine is skewed.
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


class LocalEmbedder:
    """Embed queries with a local sentence-transformers model (dev machines only)."""

    def __init__(self, model: str = EMBED_MODEL):
        from src.embeddings import EmbeddingManager
        self._mgr = EmbeddingManager(model_name=model, encode_batch_size=8)

    def embed_query(self, text: str) -> np.ndarray:
        return self._mgr.generate_embeddings([text], is_query=True)[0]


def get_embedder():
    """Pick an embedder: explicit RAG_EMBEDDER wins, else HF when a token exists.

    Defaulting to HF whenever a token is present keeps local runs honest — you exercise
    the same path the deployment will use, rather than discovering an API difference
    only after deploying.
    """
    choice = (os.getenv("RAG_EMBEDDER") or "").lower()
    if choice == "local":
        return LocalEmbedder()
    if choice == "hf":
        return HFInferenceEmbedder()
    return HFInferenceEmbedder() if hf_token() else LocalEmbedder()


# ------------------------------------------------------------------- corpus for BM25
def load_chunks_from_pinecone(store: PineconeVectorStore, batch: int = 1000) -> List[Dict[str, Any]]:
    """Pull every chunk's text + metadata back out of the index.

    The deployed app cannot ship the PDFs (licensed material), but Pinecone already holds
    each chunk's text in metadata, so the keyword index can be rebuilt from what is
    already there. Returns dicts shaped like LangChain documents so BM25Index can consume
    them unchanged.
    """
    # 1000 is Pinecone's per-fetch ceiling. At 200 this took over two minutes for 21k
    # chunks — roughly 105 network round trips; at 1000 it is 21.
    docs: List[Dict[str, Any]] = []
    for page in store.index.list(namespace=store.namespace):
        # index.list() yields ListResponse objects holding ListItem, not plain strings.
        # Slicing one directly raises KeyError: slice(0, 200) — flatten to str ids first.
        ids = [getattr(item, "id", item) for item in page]
        for start in range(0, len(ids), batch):
            fetched = store.index.fetch(ids=ids[start:start + batch], namespace=store.namespace)
            for rec in (fetched.vectors or {}).values():
                meta = dict(rec.metadata or {})
                text = meta.pop(TEXT_KEY, "")
                if text:
                    docs.append({"page_content": text, "metadata": meta})
    return docs


class _Doc:
    """Minimal duck-typed document so BM25Index sees .page_content / .metadata."""
    __slots__ = ("page_content", "metadata")

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


def build_bm25(docs: List[Dict[str, Any]]) -> BM25Index:
    return BM25Index().build([_Doc(d["page_content"], d["metadata"]) for d in docs])




# --------------------------------------------------------------------- data locations
# This repo deliberately ships no corpus and no vector store. Both are large (524 MB) and
# already committed to the NeuroApp repository through Git LFS, whose free tier allows
# 1 GB of storage — a third copy would exceed it. Paths therefore resolve in order:
# an explicit environment variable, then a local folder, then a sibling NeuroApp checkout.
def _resolve_data_path(env_var: str, local: str, sibling: str, marker: str = "") -> str:
    """Resolve a data location, testing for a MARKER file rather than the directory.

    Chromadb creates an empty store directory on a failed open, so a bare `.exists()` on
    the folder reports success for a directory that holds nothing — and the caller then
    fails with "Collection does not exist" instead of falling through to the sibling.
    """
    explicit = os.getenv(env_var)
    if explicit:
        return explicit
    if (Path(local) / marker if marker else Path(local)).exists():
        return local
    alt = Path(__file__).resolve().parent.parent.parent / sibling
    if (alt / marker if marker else alt).exists():
        return str(alt)
    return local


CHROMA_PATH = _resolve_data_path("RAG_CHROMA_PATH", "data/vector_store",
                                 "NeuroApp/data/vector_store", marker="chroma.sqlite3")
PDF_DIR = _resolve_data_path("RAG_PDF_DIR", "pdfs", "NeuroApp/pdfs")

# --------------------------------------------------------------------------- backends
# A vector store is never usable on its own: the query embedder must be the model that
# built it, the abstention floor is calibrated against that model's score distribution,
# and the BM25 arm must be chunked the same way. Bundling the four together is what makes
# falling back from one store to another safe — swap the store alone and you get a
# dimension error at best, meaningless neighbours at worst.
@dataclass
class Backend:
    name: str
    store: Any                 # exposes .query(vec, n_results) and .count()
    embedder: Any              # exposes .embed_query(text)
    min_cosine: float          # calibrated on THIS store's score distribution
    chunk_size: int            # so a BM25 index can be built to match
    chunk_overlap: int


class ChromaVectorStore:
    """Local Chroma collection behind the same surface as PineconeVectorStore.

    Chroma returns a cosine DISTANCE; Pinecone returns a cosine SIMILARITY. Converting
    here means callers never have to remember which convention they hold — getting that
    backwards inverts the ranking while every number still looks plausible.
    """

    def __init__(self, path: str = None, collection: str = "pdf_documents"):
        path = path or CHROMA_PATH
        import chromadb
        self.collection_name = collection
        self.index_name = f"chroma:{collection}"
        self._col = chromadb.PersistentClient(path=path).get_collection(collection)

    def count(self) -> int:
        return self._col.count()

    def query(self, embedding, n_results: int = 6) -> List[Dict[str, Any]]:
        vec = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        res = self._col.query(query_embeddings=[vec], n_results=max(1, int(n_results)))
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0] or [{}] * len(docs)
        dists = res.get("distances", [[]])[0] or [None] * len(docs)
        hits = [{"content": d, "metadata": dict(m or {}),
                 "similarity": (1 - dist) if dist is not None else None}
                for d, m, dist in zip(docs, metas, dists)]
        hits.sort(key=lambda h: (h["similarity"] is not None, h["similarity"]), reverse=True)
        return hits


def pinecone_backend() -> Backend:
    """BGE-M3 / Pinecone. Floor 0.53 from the measured separation:
    in-corpus min 0.573, out-of-corpus max 0.493."""
    return Backend("pinecone", get_store(), get_embedder(),
                   min_cosine=float(os.getenv("RAG_MIN_COSINE", "0.53")),
                   chunk_size=1500, chunk_overlap=300)


def chroma_backend() -> Backend:
    """all-MiniLM-L6-v2 / local Chroma. Floor 0.43 from the measured separation:
    in-corpus min 0.485 (tofersen SOD1 ALS), out-of-corpus max 0.378 (sourdough)."""
    return Backend("chroma", ChromaVectorStore(), LocalEmbedder("all-MiniLM-L6-v2"),
                   min_cosine=float(os.getenv("RAG_CHROMA_MIN_COSINE", "0.43")),
                   chunk_size=1000, chunk_overlap=200)


def _backend_dim(backend: "Backend") -> int:
    return 384 if backend.name == "chroma" else EMBED_DIM


def get_backend(prefer: str = None) -> Backend:
    """Preferred backend, falling back to the local one when it is unreachable.

    Pinecone's free tier meters egress (1 GB/month) and returns 429 once spent — at which
    point even a six-record query fails. The local Chroma store has no quota and no
    network, so it keeps the app answering; its numbers are simply not the ones the
    evaluation recorded, which is why the backend name is surfaced in the UI.
    """
    prefer = (prefer or os.getenv("RAG_BACKEND") or "pinecone").lower()
    order = [prefer] + [b for b in ("pinecone", "chroma") if b != prefer]
    errors = []
    for name in order:
        try:
            backend = pinecone_backend() if name == "pinecone" else chroma_backend()
            # Probe the operation we actually depend on. count() goes through
            # describe_index_stats, which is NOT metered as egress — it answers happily
            # while every real query 429s, so probing with it selects a dead backend.
            backend.store.query([0.0] * _backend_dim(backend), n_results=1)
            return backend
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__} {str(exc)[:110]}")
    raise RuntimeError("No vector backend available:\n  " + "\n  ".join(errors))


def build_bm25_local(chunk_size: int, chunk_overlap: int, pdf_dir: str = None) -> BM25Index:
    """Build the keyword index from the local PDF cache rather than the vector store.

    Pulling 21k chunks back out of Pinecone cost ~250 MB of egress per rebuild — each
    record ships its 1024 floats alongside the text, and `fetch` has no way to exclude
    them — which exhausted the free tier's monthly gigabyte. The corpus is already on disk
    and re-splitting it takes seconds, so `load_chunks_from_pinecone` is now only for a
    deployed container that has no PDFs.
    """
    from src import process_all_pdfs, split_documents
    chunks = split_documents(process_all_pdfs(pdf_dir or PDF_DIR, True),
                             chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return BM25Index().build(chunks)


# ------------------------------------------------------------------------- retrieval
class Retriever:
    """Hybrid retrieval over one Pinecone index and one in-memory BM25 index."""

    def __init__(self, store, embedder, bm25: Optional[BM25Index] = None,
                 min_cosine: float = MIN_COSINE):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        # Carried per-instance, not read from the module: each backend's floor is
        # calibrated against its own embedding model's score distribution.
        self.min_cosine = min_cosine

    @classmethod
    def from_backend(cls, backend: "Backend", bm25: Optional[BM25Index] = None) -> "Retriever":
        return cls(backend.store, backend.embedder, bm25, min_cosine=backend.min_cosine)

    # -- arms ---------------------------------------------------------------
    def _vector_search(self, query: str, n: int) -> List[Dict[str, Any]]:
        """Nearest n DISTINCT chunks. `similarity` is Pinecone's cosine score used as
        returned — higher is closer. Chroma reported a distance and needed `1 - dist`;
        applying that here would invert the ranking while the numbers still looked
        plausible."""
        q = self.embedder.embed_query(query)
        total = self.store.count() or (n * 3)
        ceiling = min(total, MAX_TOP_K)
        fetch = min(n * 3, ceiling)
        while True:
            raw = self.store.query(q, n_results=fetch)
            hits, seen = [], set()
            for r in raw:
                key = content_key(r["content"])
                if key in seen:
                    continue
                seen.add(key)
                hits.append({**r, "rank": len(hits) + 1, "content_key": key})
                if len(hits) == n:
                    return hits
            if fetch >= ceiling:
                return hits
            fetch = min(fetch * 5, ceiling)

    def _keyword_search(self, query: str, n: int) -> List[Dict[str, Any]]:
        return self.bm25.search(query, n) if self.bm25 else []

    # -- fusion -------------------------------------------------------------
    @staticmethod
    def _fuse_rrf(vector_hits, keyword_hits, rrf_k: int = 60):
        """Reciprocal Rank Fusion: score = sum over arms of 1 / (rrf_k + rank).

        Rank, not raw score: cosine is bounded 0..1 while BM25 is unbounded and
        corpus-relative, so no fixed weighting of the two means the same thing from one
        query to the next. The consequence is that agreement outranks precision — a chunk
        at ranks 3 and 4 in both arms (1/63 + 1/64 = 0.0315) beats a chunk at rank 1 in
        one arm (1/61 = 0.0164).
        """
        fused: Dict[str, Dict[str, Any]] = {}
        for arm, hits in (("vector", vector_hits), ("keyword", keyword_hits)):
            for h in hits:
                key = h["content_key"]
                e = fused.setdefault(key, {
                    "content": h["content"], "metadata": h["metadata"],
                    "similarity": None, "bm25_score": None,
                    "vector_rank": None, "keyword_rank": None,
                    "fusion_score": 0.0, "content_key": key,
                })
                e["fusion_score"] += 1.0 / (rrf_k + h["rank"])
                if arm == "vector":
                    e["vector_rank"], e["similarity"] = h["rank"], h.get("similarity")
                else:
                    e["keyword_rank"], e["bm25_score"] = h["rank"], h.get("bm25_score")
        return sorted(fused.values(), key=lambda x: x["fusion_score"], reverse=True)

    # -- abstention ---------------------------------------------------------
    @staticmethod
    def arm_scores(vector_hits, keyword_hits):
        best_cos = max((h.get("similarity") or 0.0) for h in vector_hits) if vector_hits else 0.0
        best_bm = max((h.get("bm25_score") or 0.0) for h in keyword_hits) if keyword_hits else 0.0
        return best_cos, best_bm

    def clears_floor(self, best_cos: float) -> bool:
        """Cosine ONLY. BM25 stays a full partner in RANKING but is excluded here.

        Ranking compares scores within one query, where BM25 excels. A floor compares a
        score against a fixed constant across queries, which needs a bounded scale.
        Measured on this corpus: "How do I center a div in CSS?" scored bm25 5.18, HIGHER
        than the genuine clinical query "Therapy for anticholinergic syndrome" at 4.66,
        because words like "center" are common in medical prose. Under the old
        `cos >= MIN_COSINE or bm25 >= MIN_BM25` rule BM25 overrode a correct cosine
        rejection and the bot answered a CSS question from neurology textbooks.
        """
        return best_cos >= self.min_cosine

    # -- public -------------------------------------------------------------
    def retrieve(self, query: str, k: int = 6, mode: str = "hybrid",
                 candidate_k: int = 20, rrf_k: int = 60) -> Dict[str, Any]:
        """Return {hits, abstained, best_cosine, best_bm25, mode}."""
        n = candidate_k if mode == "hybrid" else k
        vector_hits = self._vector_search(query, n) if mode in ("vector", "hybrid") else []
        keyword_hits = self._keyword_search(query, n) if mode in ("keyword", "hybrid") else []

        best_cos, best_bm = self.arm_scores(vector_hits, keyword_hits)
        # The floor reads the vector arm, so consult it even in keyword-only mode —
        # otherwise keyword mode could never abstain and would answer anything.
        if mode == "keyword":
            best_cos, _ = self.arm_scores(self._vector_search(query, 1), [])

        if not self.clears_floor(best_cos):
            return {"hits": [], "abstained": True, "best_cosine": best_cos,
                    "best_bm25": best_bm, "mode": mode}

        if mode == "vector":
            hits = vector_hits[:k]
        elif mode == "keyword":
            hits = keyword_hits[:k]
        else:
            hits = self._fuse_rrf(vector_hits, keyword_hits, rrf_k=rrf_k)[:k]
        return {"hits": hits, "abstained": False, "best_cosine": best_cos,
                "best_bm25": best_bm, "mode": mode}


# ------------------------------------------------------------------------ generation
REFUSAL = (
    "I could not find this in the source documents, so I am not answering. "
    "These sources are neurology and internal-medicine references."
)

INSTRUCTIONS = """You are a helpful assistant who is good at analyzing source information and answering questions.       Use the following source documents to answer the user's questions.       If you don't know the answer, just say that you don't know.       Use three sentences maximum and keep the answer concise.

Documents:
{docs}"""
# Verbatim from the evaluated notebook, whitespace included. Every score on record —
# correctness 0.56 in-corpus, 0.90 for keyword retrieval — was produced by exactly this
# text. A "better" prompt here would be unmeasured, and the numbers would stop describing
# what the app actually does. Change it only alongside a re-run.


def make_rag_bot(llm, retriever: Retriever, mode: str = "hybrid", k: int = 6):
    """retrieve -> (abstain | generate). Returns {answer, documents, abstained, ...}."""

    def rag_bot(question: str) -> Dict[str, Any]:
        res = retriever.retrieve(question, k=k, mode=mode)
        if res["abstained"]:
            return {"answer": REFUSAL, "documents": [], "abstained": True,
                    "best_cosine": res["best_cosine"], "best_bm25": res["best_bm25"],
                    "retrieval_mode": mode}
        docs_string = "\n\n".join(h["content"] for h in res["hits"])
        msg = llm.invoke([
            {"role": "system", "content": INSTRUCTIONS.format(docs=docs_string)},
            {"role": "user", "content": question},
        ])
        return {"answer": getattr(msg, "text", None) or msg.content,
                "documents": res["hits"], "abstained": False,
                "best_cosine": res["best_cosine"], "best_bm25": res["best_bm25"],
                "retrieval_mode": mode}

    return rag_bot


def get_store(index_name: str = DEFAULT_INDEX) -> PineconeVectorStore:
    return PineconeVectorStore(index_name=index_name, dimension=EMBED_DIM)
