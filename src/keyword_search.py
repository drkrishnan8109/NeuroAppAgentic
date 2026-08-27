"""BM25 keyword search over the same document chunks used by the vector store.

Why this exists
---------------
The Chroma vector store finds chunks that are *semantically* near the query. That is
exactly wrong for rare literal tokens — drug names, gene symbols, criteria names,
dosages ("tofersen", "SOD1", "El Escorial", "1 g x 3-5 d"). An embedding blurs those
into nearby concepts; BM25 matches them as strings. Running both and fusing the
results ("hybrid search") recovers what each arm alone misses.

BM25 in one paragraph
---------------------
BM25 scores a chunk against a query by summing, over the query terms it contains:
  * term frequency, with saturation — the 10th occurrence of a word adds far less
    than the 2nd (controlled by `k1`);
  * inverse document frequency — a term appearing in few chunks is worth more than
    one appearing everywhere, so "the" contributes nothing and "tofersen" a lot;
  * length normalisation — long chunks are penalised so they don't win merely by
    containing more words (controlled by `b`).
Scores are unbounded and corpus-relative: comparable *within* one query's result
list, meaningless across queries or across corpora. That is the reason the hybrid
retriever fuses by RANK (RRF) rather than by raw score.

Engine
------
`bm25s` — an in-process BM25 over scipy sparse matrices. No server, no JVM. At this
corpus size (~11k chunks) a full-blown Lucene service (Elasticsearch/OpenSearch)
would add operational weight for no gain; those matter at millions of documents or
when several applications share one index.

Typical use:

    from src import BM25Index

    bm25 = BM25Index()
    bm25.build(chunks)                      # same `chunks` list that feeds VectorStore
    hits = bm25.search("anticoagulation after stroke", k=6)
"""

import hashlib
import re
from typing import Any, Dict, List, Optional

import bm25s


def content_key(text: str) -> str:
    """Stable identifier for a chunk, derived from its text.

    Used as the join key when fusing vector and keyword hits. Deliberately NOT the
    Chroma id: `VectorStore.add_documents` mints a fresh uuid per call, so the same
    chunk carries a different id on every re-run and an id-based join would silently
    match nothing. Whitespace is collapsed first so trivial formatting differences
    don't split one chunk into two keys.
    """
    normalised = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


class BM25Index:
    """In-memory BM25 index over LangChain document chunks.

    The index is rebuilt from `chunks` at each kernel start rather than persisted:
    indexing ~11k chunks takes a couple of seconds, and keeping it in memory removes
    a whole class of staleness bugs where the keyword index and the vector store
    disagree about what the corpus contains.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        method: str = "lucene",
        stopwords: str = "en",
    ):
        """
        Args:
            k1: Term-frequency saturation. Higher = repeated terms keep adding score.
            b: Length normalisation, 0..1. 0.75 is the standard compromise.
            method: BM25 variant. "lucene" matches what Elasticsearch/Lucene compute,
                which makes results comparable to a production search engine.
            stopwords: Stopword list passed to the tokenizer. Must be identical at
                index time and query time or terms fail to align.
        """
        self.k1 = k1
        self.b = b
        self.method = method
        self.stopwords = stopwords

        self.retriever: Optional[bm25s.BM25] = None
        self.documents: List[Any] = []
        self.corpus_size = 0

    def build(self, documents: List[Any]) -> "BM25Index":
        """Tokenise and index a list of LangChain Documents.

        Returns self so it can be chained: `bm25 = BM25Index().build(chunks)`.
        """
        if not documents:
            raise ValueError("Cannot build a BM25 index from an empty document list.")

        self.documents = list(documents)
        self.corpus_size = len(self.documents)

        print(f"Building BM25 index over {self.corpus_size} chunks...")
        texts = [d.page_content for d in self.documents]

        # show_progress=False keeps notebook output readable; the whole build is fast.
        corpus_tokens = bm25s.tokenize(
            texts, stopwords=self.stopwords, show_progress=False
        )

        self.retriever = bm25s.BM25(k1=self.k1, b=self.b, method=self.method)
        self.retriever.index(corpus_tokens, show_progress=False)

        print(f"BM25 index ready ({self.method} variant, k1={self.k1}, b={self.b}).")
        return self

    def search(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        """Return the top-k chunks for `query`, best first.

        Each hit mirrors the shape the vector retriever returns, so downstream code
        (fusion, the generation prompt, the LangSmith evaluators) can treat hits from
        either arm identically.

        Fewer than k hits come back when the corpus holds fewer real matches: bm25s
        pads its top-k with zero-score chunks that share no term with the query, and
        those are dropped here. Keeping them would be actively harmful — a padded
        chunk still occupies a rank, and rank is exactly what RRF fusion rewards, so
        a one-rare-word query would hand undeserved credit to arbitrary chunks.
        """
        if self.retriever is None:
            raise RuntimeError("BM25 index not built — call build(chunks) first.")

        # The query MUST go through the same tokenizer as the corpus. return_ids=False
        # yields plain token strings, which bm25s maps against the index vocabulary and
        # which lets unseen query terms drop out harmlessly instead of erroring.
        query_tokens = bm25s.tokenize(
            query, stopwords=self.stopwords, return_ids=False, show_progress=False
        )

        # bm25s raises if k exceeds the corpus size; clamp rather than surprise the caller.
        k = max(1, min(k, self.corpus_size))

        indices, scores = self.retriever.retrieve(
            query_tokens, k=k, show_progress=False
        )

        hits: List[Dict[str, Any]] = []
        for doc_i, score in zip(indices[0], scores[0]):
            if float(score) <= 0.0:
                continue  # padding, not a match — see the docstring
            doc = self.documents[int(doc_i)]
            rank = len(hits) + 1
            hits.append(
                {
                    "content": doc.page_content,
                    "metadata": dict(doc.metadata),
                    # Unbounded, corpus-relative. Read it as "how strongly this chunk
                    # matched the query's rare terms", comparable only against the other
                    # scores in this same list.
                    "bm25_score": float(score),
                    "rank": rank,
                    "chunk_index": int(doc_i),
                    "content_key": content_key(doc.page_content),
                }
            )
        return hits
