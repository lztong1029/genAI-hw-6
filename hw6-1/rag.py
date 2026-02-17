# rag.py
from __future__ import annotations

import os
import glob
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Dict

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from bs4 import BeautifulSoup


@dataclass
class Chunk:
    text: str
    source: str  # filename or path
    idx: int     # chunk index within file


def read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    if ext in [".html", ".htm"]:
        soup = BeautifulSoup(raw, "html.parser")
        return soup.get_text(separator="\n")
    return raw


def simple_chunk(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    A simple character-based chunker (good enough for HW).
    chunk_size/overlap can be adjusted in critique.
    """
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


class VectorRAG:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "hw6_1"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._embedding_fn = None
        self._client = None
        self._col = None

    @property
    def embedding_fn(self):
        """Lazy initialization of embedding function to avoid PyTorch meta tensor issues."""
        if self._embedding_fn is None:
            # Set environment variable to avoid tokenizer parallelism warnings
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            self._embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        return self._embedding_fn

    @property
    def client(self):
        """Lazy initialization of ChromaDB client."""
        if self._client is None:
            # Clean up corrupted database if exists
            if os.path.exists(self.persist_dir):
                try:
                    # Try to create client first
                    self._client = chromadb.PersistentClient(path=self.persist_dir)
                except (ValueError, Exception):
                    # If failed, remove corrupted database and recreate
                    shutil.rmtree(self.persist_dir, ignore_errors=True)
                    self._client = chromadb.PersistentClient(path=self.persist_dir)
            else:
                self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    @property
    def col(self):
        """Lazy initialization of collection."""
        if self._col is None:
            self._col = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )
        return self._col

    def index_folder(self, data_dir: str, exts=(".md", ".txt", ".html", ".htm", ".json", ".csv")) -> int:
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(data_dir, f"**/*{ext}"), recursive=True))

        chunks: List[Chunk] = []
        for p in sorted(paths):
            content = read_file(p)
            for i, c in enumerate(simple_chunk(content)):
                chunks.append(Chunk(text=c, source=os.path.relpath(p, data_dir), idx=i))

        if not chunks:
            return 0

        # Upsert into Chroma
        ids = [f"{c.source}::chunk{c.idx}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [{"source": c.source, "chunk": c.idx} for c in chunks]

        # Avoid duplicating if re-run: delete then add (simple strategy)
        # For small HW datasets this is fine.
        try:
            self.col.delete(where={})
        except Exception:
            pass

        self.col.add(ids=ids, documents=documents, metadatas=metadatas)
        return len(chunks)

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        res = self.col.query(query_texts=[query], n_results=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        hits = []
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({
                "text": doc,
                "source": meta.get("source"),
                "chunk": meta.get("chunk"),
                "distance": dist,
            })
        return hits

    def naive_answer(self, query: str, hits: List[Dict]) -> str:
        """
        Minimal 'answer' without calling an LLM:
        we just summarize what was retrieved in a deterministic way.
        For HW, you can later replace this with an actual LLM call.
        """
        if not hits:
            return "No relevant context found in your documents."

        # Very simple: show top contexts and let user read.
        lines = ["Top retrieved context (read these as supporting evidence):"]
        for i, h in enumerate(hits, 1):
            snippet = h["text"][:280].replace("\n", " ")
            lines.append(f"{i}. [{h['source']}#chunk{h['chunk']}] {snippet}...")
        return "\n".join(lines)