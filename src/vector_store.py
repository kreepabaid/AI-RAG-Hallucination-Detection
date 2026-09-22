"""
vector_store.py
Embeds document chunks with sentence-transformers and stores in FAISS.
"""

import numpy as np
import faiss
import torch
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class VectorStore:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading embedding model '{model_name}' on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.chunks: List[Dict] = []
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        elif hasattr(self.model, "get_sentence_embedding_dimension"):
            self.dimension = self.model.get_sentence_embedding_dimension()
        else:
            self.dimension = 384

    def add_chunks(self, chunks: List[Dict]):
        """Embed chunks and add to FAISS index."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            batch_size=32,
            device=self.device
        )
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Find top-k most relevant chunks for a query."""
        if self.index is None or not self.chunks or not query.strip():
            return []

        k = max(1, min(top_k, len(self.chunks)))
        q_emb = self.model.encode([query], device=self.device)
        q_emb = np.array(q_emb, dtype="float32")
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

    def clear(self):
        """Reset index and stored chunks."""
        self.index = None
        self.chunks = []

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def get_sources(self) -> List[str]:
        return list(dict.fromkeys(c.get("source", "Unknown") for c in self.chunks))
