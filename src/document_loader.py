"""Reusable PDF loading and text-splitting utilities for the RAG pipeline.

Typical use:

    from src import process_all_pdfs, split_documents

    docs   = process_all_pdfs("/path/to/pdfs")
    chunks = split_documents(docs, chunk_size=3000, chunk_overlap=500)
"""

import hashlib
import pickle
from pathlib import Path
from typing import List

# PyMuPDFLoader (fitz) rather than PyPDFLoader: pypdf trips a "Limit reached while
# decompressing" safeguard on some streams in large textbook PDFs, and is markedly
# slower (pure Python vs. C). PyMuPDF loads the same PDFs without that limit.
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Anchor the default cache location to the project root (the parent of this src/ dir)
# via __file__, rather than a cwd-relative "../data" path. A module can be imported
# from a notebook, a script, or a test — each with a different working directory — so
# a relative path would resolve inconsistently. __file__ always resolves the same way.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "pdf_documents_cache.pkl"


def _pdf_directory_fingerprint(pdf_files: List[Path]) -> str:
    """Fingerprint based on filename+size+mtime so cache invalidates only when PDFs actually change"""
    parts = [f"{f.name}:{f.stat().st_size}:{f.stat().st_mtime}" for f in sorted(pdf_files)]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def process_all_pdfs(pdf_directory, use_cache: bool = True, cache_path=DEFAULT_CACHE_PATH):
    """Process all PDF files in a directory, caching results to disk to skip re-parsing on reruns

    Args:
        pdf_directory: Directory to scan recursively for *.pdf files
        use_cache: When True, load from / save to the pickle cache at cache_path
        cache_path: Where the parsed-document cache lives (defaults to <project>/data/pdf_documents_cache.pkl)

    Returns:
        List of LangChain Documents (one per PDF page), each tagged with source_file/file_type metadata
    """
    cache_path = Path(cache_path)
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    fingerprint = _pdf_directory_fingerprint(pdf_files)

    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("fingerprint") == fingerprint:
            print(f"Loaded {len(cached['documents'])} pages from cache: {cache_path}")
            return cached["documents"]
        print("PDF directory changed since last cache — reprocessing")

    all_documents = []
    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()

            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'

            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"fingerprint": fingerprint, "documents": all_documents}, f)
        print(f"Cached parsed documents to {cache_path}")

    return all_documents


def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    # Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content:  {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs
