"""Embedding generation and ChromaDB vector-store management for the RAG pipeline.

EmbeddingManager handles both:
  * simple symmetric models (e.g. all-MiniLM-L6-v2) — the default, and
  * large asymmetric models (e.g. nvidia/Nemotron-3-Embed-1B-BF16) that need
    "query:"/"passage:" prefixes, a bigger device, and bfloat16 to run well.

Typical use:

    from src import EmbeddingManager, VectorStore

    # Default: all-MiniLM-L6-v2 (384-dim, fast, no prefixes needed)
    embedding_manager = EmbeddingManager()
    vectorstore       = VectorStore()

    texts      = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)   # is_query=False -> "passage"
    vectorstore.add_documents(chunks, embeddings)
"""

import hashlib
import os
import re
from pathlib import Path
from typing import Any, List

import numpy as np
import chromadb

# sentence_transformers is imported lazily inside _load_model rather than here. It pulls
# torch (~1 GB), which does not fit a Streamlit Community Cloud container — and a deployed
# app never needs it, because queries are embedded through the HuggingFace Inference API.
# Importing it at module scope would make `from src import ...` fail on any slim install,
# even for code paths that never touch a local model.

# Anchor the default vector-store location to the project root (the parent of this src/
# dir) via __file__, so it resolves the same whether a notebook/script is launched from
# the project root or elsewhere — otherwise a cwd-relative "./data/..." path would create
# separate stores depending on where you ran from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PERSIST_DIRECTORY = str(PROJECT_ROOT / "data" / "vector_store")


# Known asymmetric models that need distinct query/passage prefixes for retrieval to
# work correctly. When one of these is used, EmbeddingManager applies the prefixes
# automatically — callers don't have to know the convention. Add new models here as
# you adopt them; anything not listed defaults to no prefix (correct for symmetric
# models like all-MiniLM-L6-v2).
_MODEL_PREFIXES = {
    "nvidia/Nemotron-3-Embed-1B-BF16": {"query": "query: ", "passage": "passage: "},
}


class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer.

    The model is chosen at construction time via `model_name`, so downstream code
    can decide which model to use. all-MiniLM-L6-v2 is the default.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = None,
        query_prefix: str = None,
        passage_prefix: str = None,
        use_bf16: bool = False,
        encode_batch_size: int = 32,
    ):
        """
        Initialize the embedding manager

        Args:
            model_name: HuggingFace model name for sentence embeddings. Anything
                SentenceTransformer can load works; all-MiniLM-L6-v2 by default.
            device: torch device ('mps', 'cuda', or 'cpu'). Auto-detected if None.
            query_prefix / passage_prefix: text prepended to search queries vs. document
                chunks for asymmetric models. If left None, they're looked up for known
                models (see _MODEL_PREFIXES) and otherwise default to "" (no prefix).
                Pass explicitly to override.
            use_bf16: load the model in bfloat16 with SDPA attention (recommended for
                large models like Nemotron; falls back to default dtype if unsupported).
                Leave False for small models like all-MiniLM — bf16 gives them no benefit.
            encode_batch_size: batch size passed to model.encode(). 32 is fine for small
                models; drop it (e.g. 8) for billion-parameter models to fit in memory.
        """
        self.model_name = model_name
        self.device = device or self._detect_device()

        # Prefixes: explicit args win; else fall back to the known-model registry; else none.
        known = _MODEL_PREFIXES.get(model_name, {"query": "", "passage": ""})
        self.query_prefix = query_prefix if query_prefix is not None else known["query"]
        self.passage_prefix = passage_prefix if passage_prefix is not None else known["passage"]

        self.use_bf16 = use_bf16
        self.encode_batch_size = encode_batch_size
        self.model = None
        self._load_model()

    @staticmethod
    def _detect_device() -> str:
        """Pick the fastest available torch device: Apple GPU (mps) > CUDA > CPU."""
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _embedding_dimension(self) -> int:
        """Read the output dimension, tolerating the get_sentence_embedding_dimension ->
        get_embedding_dimension rename across sentence-transformers versions."""
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def _load_model(self):
        """Load the SentenceTransformer model onto the chosen device.

        With use_bf16=True, tries bfloat16 + SDPA attention (recommended for large
        models); if that combination isn't supported on this device (Apple's MPS
        backend has had inconsistent bf16 support) it falls back to the default dtype.
        """
        from sentence_transformers import SentenceTransformer   # lazy: see module header

        print(f"Loading embedding model: {self.model_name} (device={self.device}, bf16={self.use_bf16})")
        try:
            if self.use_bf16:
                import torch
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"},
                )
            else:
                self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            print(f"Preferred load failed ({e}); falling back to default dtype/settings")
            self.model = SentenceTransformer(self.model_name, device=self.device)

        print(f"Model loaded successfully. Embedding dimension: {self._embedding_dimension()}")

        # Diagnostic: confirm the model actually landed on the requested device/dtype.
        # If this prints "cpu"/"float32" when you asked for mps/bf16, a silent fallback
        # happened and embedding will be far slower than expected.
        try:
            first_param = next(self.model[0].auto_model.parameters())
            print(f"Actual model device: {first_param.device}, dtype: {first_param.dtype}")
        except Exception:
            pass  # introspection is best-effort; not all models expose [0].auto_model

    def generate_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed
            is_query: True when embedding a search query, False when embedding document
                chunks — selects the query vs. passage prefix. Ignored (no-op) for models
                with empty prefixes, so this stays backward-compatible with symmetric models.

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")

        prefix = self.query_prefix if is_query else self.passage_prefix
        prefixed_texts = [prefix + t for t in texts] if prefix else list(texts)

        kind = "query" if is_query else "passage"
        print(f"Generating embeddings for {len(texts)} texts (as {kind})...")
        # normalize_embeddings=True: these models are compared via cosine similarity,
        # so vectors need unit length for cosine distance in the vector store to be meaningful
        embeddings = self.model.encode(
            prefixed_texts,
            show_progress_bar=True,
            batch_size=self.encode_batch_size,
            normalize_embeddings=True,
        )
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = DEFAULT_PERSIST_DIRECTORY):
        """
        Initialize the vector store

        Note: a ChromaDB collection is fixed to one embedding dimension. If you switch
        embedding models (e.g. all-MiniLM's 384-dim -> Nemotron's 2048-dim), use a
        different collection_name/persist_directory so the vectors don't collide.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store. Defaults to
                <project>/data/vector_store (anchored to the package location, not the
                current working directory).
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection with cosine distance.

        Chroma defaults new collections to L2 distance, but sentence-transformer
        embeddings are trained/compared via cosine similarity — mismatching the two
        skews which chunks rank as "closest". If an existing collection was built
        under the old default, it's recreated here (its distances were meaningless
        anyway) so add_documents needs to be re-run to repopulate it.
        """
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            existing_names = {c.name for c in self.client.list_collections()}
            if self.collection_name in existing_names:
                existing_collection = self.client.get_collection(self.collection_name)
                space = (existing_collection.metadata or {}).get("hnsw:space")
                if space != "cosine":
                    print(f"Existing collection '{self.collection_name}' uses '{space or 'l2 (default)'}' "
                          f"distance — recreating it with cosine distance (re-run add_documents to repopulate)")
                    self.client.delete_collection(self.collection_name)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG", "hnsw:space": "cosine"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store.

        Idempotent: re-running this with the same documents overwrites the existing
        records instead of appending copies. Two things make that work — a
        content-derived id (see `_document_id`) and `upsert` rather than `add`.

        Previously the id was `f"doc_{uuid.uuid4().hex[:8]}_{i}"`. Chroma deduplicates
        by id and nothing else — it never inspects content — so a fresh uuid on every
        call meant every re-run of the embedding cell appended a full second copy of
        the corpus. Twenty runs left ~224k records for ~11k distinct chunks, which
        crowded the top-k with the same chunk repeated and, once BM25 was added, would
        also have skewed document frequencies and corrupted its IDF weighting.

        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        seen_ids = set()
        within_batch_dupes = 0

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Content-derived ID -> the same chunk always maps to the same record.
            doc_id = self._document_id(doc)

            # Chroma rejects a single call containing repeated ids, so collapse
            # duplicates here. These are genuine repeats within `documents` (identical
            # text on the same source page), not an artefact of re-running.
            if doc_id in seen_ids:
                within_batch_dupes += 1
                continue
            seen_ids.add(doc_id)
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # Document content
            documents_text.append(doc.page_content)

            # Embedding
            embeddings_list.append(embedding.tolist())

        if within_batch_dupes:
            print(f"Collapsed {within_batch_dupes} duplicate chunks (identical text) "
                  f"-> {len(ids)} distinct records.")

        # Write in batches (Chroma caps how many records a single call accepts).
        # upsert, not add: with content-derived ids, re-running overwrites each record
        # in place rather than appending a duplicate.
        max_batch_size = self.client.get_max_batch_size()
        try:
            for start in range(0, len(ids), max_batch_size):
                end = start + max_batch_size
                self.collection.upsert(
                    ids=ids[start:end],
                    embeddings=embeddings_list[start:end],
                    metadatas=metadatas[start:end],
                    documents=documents_text[start:end]
                )
            print(f"Successfully upserted {len(ids)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    @staticmethod
    def _document_id(doc: Any) -> str:
        """Stable, content-derived id for a chunk.

        Keyed on source file + page + normalised text, so the same chunk yields the
        same id on every run (making writes idempotent) while the SAME text appearing
        in two different textbooks stays two records — provenance is worth keeping in
        a medical corpus, where citing the right source matters. Whitespace is
        collapsed first so trivial re-formatting doesn't mint a new record.

        Mirrors `src.keyword_search.content_key`, which keys on text alone because its
        job is the opposite: collapsing identical text across arms during fusion.
        """
        meta = getattr(doc, "metadata", {}) or {}
        source = str(meta.get("source_file", ""))
        page = str(meta.get("page", ""))
        text = re.sub(r"\s+", " ", doc.page_content).strip()
        return hashlib.sha1(f"{source}|{page}|{text}".encode("utf-8")).hexdigest()
