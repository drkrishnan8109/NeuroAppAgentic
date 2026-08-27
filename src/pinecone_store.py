"""Pinecone-backed vector store for the hybrid RAG notebook.

Why this exists alongside `src.embeddings.VectorStore`
------------------------------------------------------
`VectorStore` (ChromaDB, on-disk) still backs agentic_rag.ipynb and the other
notebooks. Only RAG_hybrid.ipynb moves to Pinecone, so the two live side by side
rather than one replacing the other. This class deliberately mirrors the Chroma
class's surface — `add_documents(documents, embeddings)`, `count()`, `query()` —
so the notebook's retrieval code changes as little as possible.

Ids are the SAME content-derived SHA-1 the Chroma store uses (reused, not
reimplemented, from `VectorStore._document_id`): source_file + page + normalised
text. Pinecone upserts by id exactly as Chroma does, so re-running the embed cell
overwrites each record in place instead of appending a duplicate corpus.

Six ways Pinecone differs from Chroma, all handled here
-------------------------------------------------------
1. SCORE SEMANTICS. Chroma returns cosine *distance* (so the notebook computed
   `similarity = 1 - dist`). Pinecone with metric="cosine" returns cosine
   *similarity* directly in `match.score`. Applying `1 - x` to it would silently
   INVERT the ranking — the results still look plausible, which is what makes it
   dangerous. `query()` returns a ready-to-use `similarity` so no caller has to
   remember which convention it is holding.
2. NO SEPARATE TEXT FIELD. Chroma stores `documents` apart from `metadatas`;
   Pinecone has only metadata, so chunk text is carried in metadata[TEXT_KEY].
   Pinecone caps metadata at 40 KB per vector — fine for ~1 KB chunks, but
   oversized chunks are skipped loudly rather than failing the whole batch.
3. EVENTUALLY CONSISTENT COUNTS. describe_index_stats() lags an upsert by
   seconds. The notebook's adaptive over-fetch loop uses the count to decide when
   it has exhausted the corpus, and a stale 0 would end that loop early, so
   `count()` takes a `settle` option that waits for the number to stop moving.
4. top_k CEILING. Pinecone allows top_k up to 10_000, but a response carrying
   metadata is capped at ~4 MB, and the notebook's over-fetch multiplies its
   window by 5 each pass. MAX_TOP_K keeps that from ever requesting a response
   the API will reject.
5. ASYNC INDEX CREATION. A just-created index is not immediately queryable, so
   creation polls `status.ready` instead of returning straight away.
6. NO LOCAL FILE. There is no on-disk artefact to inspect or delete; the index
   lives in Pinecone's cloud and `delete_all()` is the equivalent of removing the
   store directory.

Typical use:

    from src import EmbeddingManager, PineconeVectorStore

    embedding_manager = EmbeddingManager()
    vectorstore       = PineconeVectorStore()      # creates the index on first use

    texts      = [d.page_content for d in chunks]
    vectorstore.add_documents(chunks, embedding_manager.generate_embeddings(texts))
"""

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from pinecone import Pinecone, ServerlessSpec

from src.embeddings import VectorStore

# Chunk text rides in metadata under this key (see difference #2 above).
TEXT_KEY = "text"

# Pinecone's own ceiling is 10_000, but a response including metadata is limited to
# ~4 MB. At ~1 KB of text per chunk, 1000 matches is comfortably inside that while
# still being deeper than any over-fetch this pipeline performs (see difference #4).
MAX_TOP_K = 1000

# Pinecone rejects a single vector whose metadata exceeds 40 KB.
MAX_METADATA_BYTES = 40_000

# The free Starter tier serves serverless indexes from AWS us-east-1 ONLY. This is a
# constraint of the plan, not a preference — pointing it elsewhere fails at create time.
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"


class PineconeVectorStore:
    """Manages document embeddings in a Pinecone serverless index.

    Mirrors the interface of `src.embeddings.VectorStore` so the two are
    interchangeable from the notebook's point of view.
    """

    def __init__(
        self,
        index_name: str = "drkrag-pdf-documents",
        dimension: int = 384,
        metric: str = "cosine",
        namespace: str = "",
        api_key: Optional[str] = None,
        cloud: str = DEFAULT_CLOUD,
        region: str = DEFAULT_REGION,
    ):
        """
        Args:
            index_name: Pinecone index to use, created if absent. An index is fixed to
                one dimension, so switching embedding models (MiniLM's 384 ->
                Nemotron's 2048) needs a DIFFERENT index name, exactly as switching
                models needed a different Chroma collection.
            dimension: embedding width. 384 matches the all-MiniLM-L6-v2 default.
            metric: "cosine" to match the normalised embeddings this project produces.
            namespace: optional partition within the index. "" is the default namespace.
            api_key: PINECONE_API_KEY is read from the environment when omitted.
            cloud/region: serverless placement. Leave at the defaults on the free tier.
        """
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.namespace = namespace
        self.cloud = cloud
        self.region = region

        key = api_key or os.getenv("PINECONE_API_KEY")
        if not key:
            raise RuntimeError(
                "PINECONE_API_KEY is not set. Add it to the project-root .env "
                "(get a free Starter-tier key at https://app.pinecone.io)."
            )

        self.client = Pinecone(api_key=key)
        self.index = None
        self._initialize_store()

    def _initialize_store(self):
        """Connect to the index, creating it if it does not exist yet.

        Creation is asynchronous: Pinecone returns before the index is queryable, so
        this polls `status.ready` rather than handing back an index that would reject
        the very next upsert (difference #5).
        """
        try:
            existing = {i["name"] for i in self.client.list_indexes()}

            if self.index_name not in existing:
                print(f"Creating Pinecone index '{self.index_name}' "
                      f"(dim={self.dimension}, metric={self.metric}, {self.cloud}/{self.region})...")
                self.client.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )
                self._wait_until_ready()
            else:
                # An existing index is fixed to the dimension it was created with; a
                # mismatch would otherwise surface as an opaque upsert error later.
                desc = self.client.describe_index(self.index_name)
                if desc.dimension != self.dimension:
                    raise ValueError(
                        f"Index '{self.index_name}' has dimension {desc.dimension}, but this "
                        f"store expects {self.dimension}. Use a different index_name for a "
                        f"different embedding model."
                    )

            self.index = self.client.Index(self.index_name)
            print(f"Pinecone index ready: {self.index_name}")
            print(f"Existing vectors in index: {self.count()}")

        except Exception as e:
            print(f"Error initializing Pinecone index: {e}")
            raise

    def _wait_until_ready(self, timeout: float = 120.0, interval: float = 1.0):
        """Block until a freshly created index reports ready, or give up loudly."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.client.describe_index(self.index_name).status
            if status.get("ready"):
                return
            time.sleep(interval)
        raise TimeoutError(
            f"Pinecone index '{self.index_name}' was not ready within {timeout:.0f}s."
        )

    def count(self, settle: bool = False, timeout: float = 30.0) -> int:
        """Number of vectors in this namespace.

        Pinecone's stats are eventually consistent, so the value can lag an upsert by
        seconds (difference #3). Pass settle=True right after writing to wait for the
        number to stop changing — the notebook's over-fetch loop treats this count as
        the size of the corpus, and a stale 0 would cut that loop short.
        """
        if not settle:
            return self._raw_count()

        deadline = time.time() + timeout
        previous, stable = -1, 0
        while time.time() < deadline:
            current = self._raw_count()
            # Two identical readings in a row: the write has propagated.
            stable = stable + 1 if current == previous else 0
            if stable >= 1 and current > 0:
                return current
            previous = current
            time.sleep(1.0)
        return self._raw_count()

    def _raw_count(self) -> int:
        stats = self.index.describe_index_stats()
        namespaces = stats.get("namespaces") or {}
        if self.namespace:
            return int(namespaces.get(self.namespace, {}).get("vector_count", 0))
        return int(stats.get("total_vector_count", 0) or 0)

    def add_documents(self, documents: List[Any], embeddings: np.ndarray, batch_size: int = 100):
        """Upsert documents and their embeddings.

        Idempotent for the same reason the Chroma store is: the id is derived from the
        chunk's content (source_file + page + normalised text), and Pinecone upserts by
        id. Re-running the embed cell overwrites each record rather than appending a
        second copy of the corpus.

        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings, one row per document
            batch_size: vectors per request. 100 keeps each request well under
                Pinecone's 2 MB payload limit given ~1 KB of text metadata per vector.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to Pinecone index '{self.index_name}'...")

        vectors = []
        seen_ids = set()
        within_batch_dupes = 0
        oversized = 0

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Same content-derived id as the Chroma store, reused rather than
            # reimplemented so the two stores agree on what "the same chunk" means.
            doc_id = VectorStore._document_id(doc)

            # Genuine repeats within `documents` (identical text on the same source
            # page). Pinecone would keep only the last write for a repeated id anyway;
            # collapsing here makes the count honest and the batching predictable.
            if doc_id in seen_ids:
                within_batch_dupes += 1
                continue
            seen_ids.add(doc_id)

            metadata = {k: v for k, v in dict(doc.metadata).items() if v is not None}
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            # Pinecone has no separate documents field, so the text rides along in
            # metadata and comes back on query (difference #2).
            metadata[TEXT_KEY] = doc.page_content

            # Skip rather than let one oversized chunk fail its whole batch.
            if len(str(metadata).encode("utf-8")) > MAX_METADATA_BYTES:
                oversized += 1
                continue

            vectors.append({
                "id": doc_id,
                "values": [float(x) for x in embedding],
                "metadata": _coerce_metadata(metadata),
            })

        if within_batch_dupes:
            print(f"Collapsed {within_batch_dupes} duplicate chunks (identical text) "
                  f"-> {len(vectors)} distinct records.")
        if oversized:
            print(f"Skipped {oversized} chunks whose metadata exceeded "
                  f"{MAX_METADATA_BYTES} bytes.")

        try:
            for start in range(0, len(vectors), batch_size):
                self.index.upsert(
                    vectors=vectors[start:start + batch_size],
                    namespace=self.namespace,
                    show_progress=False,
                )
                done = min(start + batch_size, len(vectors))
                if done % (batch_size * 20) == 0 or done == len(vectors):
                    print(f"  upserted {done}/{len(vectors)}")

            print(f"Successfully upserted {len(vectors)} documents to Pinecone")
            # settle=True: stats lag the write, so without this the number printed
            # here would usually understate what was just written.
            print(f"Total vectors in index: {self.count(settle=True)}")

        except Exception as e:
            print(f"Error adding documents to Pinecone: {e}")
            raise

    def query(self, embedding, n_results: int = 6) -> List[Dict[str, Any]]:
        """Nearest `n_results` vectors, returned as flat dicts.

        `similarity` is Pinecone's cosine score used AS RETURNED — higher is closer.
        Chroma reports a distance and the notebook converted it with `1 - dist`; doing
        that here would invert the ranking (difference #1). Returning the corrected
        value from one place keeps that mistake out of every caller.
        """
        top_k = max(1, min(int(n_results), MAX_TOP_K))
        vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

        res = self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
            include_values=False,
        )

        hits = []
        for match in (res.get("matches") or []):
            metadata = dict(match.get("metadata") or {})
            # Text comes back out of metadata; the rest stays as the chunk's metadata
            # so source_file/page remain available for citations.
            content = metadata.pop(TEXT_KEY, "")
            hits.append({
                "content": content,
                "metadata": metadata,
                "similarity": match.get("score"),
            })

        # Sort by score DESCENDING rather than trusting the response order. Pinecone
        # serverless has been observed returning matches out of score order (a 3-vector
        # index came back 0.69, 0.23, 0.42), because results are merged from segments
        # without a guaranteed global sort. Callers turn position into `rank`, and RRF
        # fuses on RANK alone — so an unsorted response would quietly corrupt every
        # hybrid result while every individual score still looked right.
        hits.sort(key=lambda h: (h["similarity"] is not None, h["similarity"]), reverse=True)
        return hits

    def delete_all(self):
        """Remove every vector in this namespace — the Pinecone equivalent of deleting
        the on-disk Chroma directory. The index itself is kept."""
        self.index.delete(delete_all=True, namespace=self.namespace)
        print(f"Deleted all vectors in index '{self.index_name}' (namespace={self.namespace or 'default'})")


def _coerce_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce metadata into the value types Pinecone accepts.

    Pinecone allows string, number, boolean, and list-of-string. PyMuPDF page metadata
    carries datetimes and other objects that would be rejected, so anything outside the
    accepted set is stringified rather than dropped — provenance fields like moddate are
    worth keeping in readable form.
    """
    clean: Dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = [str(v) for v in value]
        else:
            clean[key] = str(value)
    return clean
