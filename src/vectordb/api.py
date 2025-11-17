import numpy as np
from typing import Dict, List

from .storage import InMemoryStorage
from .index import cosine_similarity
from .embeddings import Embeddings

class VectorDB:
    def __init__(self, dim: int, embeddings: Embeddings | None = None):
        self.dim = dim
        self.storage = InMemoryStorage(dim)
        self.embeddings = embeddings

    def add_vector(self, vector: np.ndarray, metadata: Dict | None = None) -> str:
        return self.storage.add(vector, metadata)

    def search_vector(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        vectors, metas, ids = self.storage.get_all()
        if vectors.shape[0] == 0:
            return []
        scores = cosine_similarity(query_vector, vectors)
        top = np.argsort(-scores)[:k]
        return [{"id": ids[i], "score": float(scores[i]), "metadata": metas[i]} for i in top]

    def add_text(self, text: str, metadata: Dict | None = None) -> str:
        if self.embeddings is None:
            raise RuntimeError("No embeddings model provided")
        vector = self.embeddings.embed(text)
        metadata = metadata.copy() if metadata else {}
        metadata["text"] = text
        return self.add_vector(vector, metadata)

    def search_text(self, query: str, k: int = 5) -> List[Dict]:
        if self.embeddings is None:
            raise RuntimeError("No embeddings model provided")
        return self.search_vector(self.embeddings.embed(query), k)
