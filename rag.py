"""Minimal RAG pipeline: load -> chunk -> embed -> retrieve."""
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class RAGStore:
    def __init__(self, docs_dir="docs", chunk_size=500, overlap=100,
                 model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.chunks = []      # list of {"text": ..., "source": ...}
        self.embeddings = None
        self._load(Path(docs_dir), chunk_size, overlap)

    # ---------- indexing ----------
    @staticmethod
    def _chunk(text, size, overlap):
        step = max(size - overlap, 1)
        return [text[i:i + size].strip()
                for i in range(0, len(text), step) if text[i:i + size].strip()]

    def _load(self, docs_dir, size, overlap):
        files = [p for p in docs_dir.rglob("*") if p.suffix in {".txt", ".md"}]
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for c in self._chunk(text, size, overlap):
                self.chunks.append({"text": c, "source": path.name})
        if not self.chunks:
            raise ValueError(f"No .txt/.md documents found in '{docs_dir}'")
        self.embeddings = self.embedder.encode(
            [c["text"] for c in self.chunks], normalize_embeddings=True
        )
        print(f"[RAG] Indexed {len(self.chunks)} chunks from {len(files)} files.")

    # ---------- retrieval ----------
    def search(self, query, top_k=3):
        q = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q            # cosine similarity
        best = np.argsort(scores)[::-1][:top_k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in best]
