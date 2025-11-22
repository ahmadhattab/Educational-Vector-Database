import pickle
from typing import Dict, List, Optional

import numpy as np

from .storage import InMemoryStorage
from .index import similarity_scores
from .embeddings import Embeddings


class VectorDB:
    """
    v2:
    - دعم metric مختلف (cosine, euclidean, dot)
    - دعم التخزين على القرص (save / load)
    """

    def __init__(
        self,
        dim: int,
        embeddings: Optional[Embeddings] = None,
        metric: str = "cosine",
    ):
        self.dim = dim
        self.storage = InMemoryStorage(dim)
        self.embeddings = embeddings
        self.metric = metric  # "cosine" أو "euclidean" أو "dot"

    # ---------------------------
    # Low-level (vectors)
    # ---------------------------

    def add_vector(self, vector: np.ndarray, metadata: Optional[Dict] = None) -> str:
        return self.storage.add(vector, metadata)

    def search_vector(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        vectors, metas, ids = self.storage.get_all()
        if vectors.shape[0] == 0:
            return []

        scores = similarity_scores(query_vector, vectors, metric=self.metric)
        top = np.argsort(-scores)[:k]

        results = []
        for i in top:
            results.append(
                {
                    "id": ids[i],
                    "score": float(scores[i]),
                    "metadata": metas[i],
                }
            )
        return results

    # ---------------------------
    # High-level (text)
    # ---------------------------

    def add_text(self, text: str, metadata: Optional[Dict] = None) -> str:
        if self.embeddings is None:
            raise RuntimeError("No embeddings model configured.")
        vector = self.embeddings.embed(text)
        meta_final = metadata.copy() if metadata else {}
        meta_final["text"] = text
        return self.add_vector(vector, meta_final)

    def search_text(self, query: str, k: int = 5) -> List[Dict]:
        if self.embeddings is None:
            raise RuntimeError("No embeddings model configured.")
        q_vec = self.embeddings.embed(query)
        return self.search_vector(q_vec, k=k)

    # ---------------------------
    # Persistence (save / load)
    # ---------------------------

    def save(self, path: str) -> None:
        """
        حفظ حالة قاعدة البيانات على القرص.
        لا نخزن موديل الـ Embeddings نفسه، فقط اسمه لو حاب.
        """
        data = {
            "dim": self.dim,
            "metric": self.metric,
            "storage": self.storage.to_dict(),
            # ممكن تحفظ اسم الموديل لو كان HFSentenceTransformer:
            "embeddings_class": self.embeddings.__class__.__name__ if self.embeddings else None,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str, embeddings: Optional[Embeddings] = None) -> "VectorDB":
        """
        تحميل قاعدة البيانات من ملف.
        مهم تمرر Embeddings من الخارج لأننا ما بنخزن الموديل نفسه داخل الملف.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        dim = data["dim"]
        metric = data.get("metric", "cosine")
        db = cls(dim=dim, embeddings=embeddings, metric=metric)
        db.storage = InMemoryStorage.from_dict(data["storage"])
        return db
