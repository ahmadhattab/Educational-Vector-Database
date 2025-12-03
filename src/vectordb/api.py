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

    v3:
    - خيار استخدام فهرس ANN (HNSW) للبحث الأسرع في المجموعات الكبيرة.
    """

    def __init__(
        self,
        dim: int,
        embeddings: Optional[Embeddings] = None,
        metric: str = "cosine",
        use_ann: bool = False,
    ):
        self.dim = dim
        self.storage = InMemoryStorage(dim)
        self.embeddings = embeddings
        self.metric = metric  # "cosine" أو "euclidean" أو "dot"
        self.use_ann = use_ann

        # ann_index يبنى عند أول إضافة أو عند الطلب
        self.ann_index = None
        self._ann_index_size = 0

    # ---------------------------
    # Low-level (vectors)
    # ---------------------------

    def _maybe_init_ann(self):
        """
        يبني فهرس HNSW من المتجهات الحالية عند الحاجة.
        """
        if not self.use_ann:
            return

        from .ann import HNSWIndex

        vectors, _, _ = self.storage.get_all()
        if vectors.shape[0] == 0:
            self.ann_index = None
            self._ann_index_size = 0
            return

        n_vectors = vectors.shape[0]
        if self.ann_index is None or self._ann_index_size != n_vectors:
            self.ann_index = HNSWIndex(dim=self.dim, metric=self.metric)
            self.ann_index.build(vectors)
            self._ann_index_size = n_vectors

    def _ann_add_single(self, vector: np.ndarray, idx: int):
        """
        إضافة متجه واحد إلى الفهرس بدون إعادة البناء الكامل.
        """
        if not self.use_ann:
            return
        if self.ann_index is None:
            # لو ما فيه فهرس، نبنيه من جديد.
            try:
                self._maybe_init_ann()
            except ImportError:
                # hnswlib غير متوفر، نرجع للبحث الخطي.
                self.use_ann = False
            return

        self.ann_index.add_items(np.array([vector]), [idx])
        self._ann_index_size = max(self._ann_index_size, idx + 1)

    def _convert_ann_distances_to_similarity(self, distances: np.ndarray) -> np.ndarray:
        """
        hnswlib يرجع distance، نحولها إلى قيمة تشابه قابلة للمقارنة
        مع البحث الخطي.
        """
        metric = self.metric.lower()
        if metric == "cosine":
            # cosine في hnswlib عبارة عن مسافة (0 = متطابق)، نحولها لتشابه.
            return 1.0 - distances
        elif metric == "euclidean":
            return 1.0 / (1.0 + distances)
        else:  # dot / ip
            return distances

    def add_vector(self, vector: np.ndarray, metadata: Optional[Dict] = None) -> str:
        _id = self.storage.add(vector, metadata)

        if self.use_ann:
            _, _, ids = self.storage.get_all()
            new_idx = len(ids) - 1  # index داخل المصفوفة
            self._ann_add_single(vector, new_idx)

        return _id

    def search_vector(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        vectors, metas, ids = self.storage.get_all()
        if vectors.shape[0] == 0:
            return []

        # -------------------------
        # خيار ANN باستخدام HNSW
        # -------------------------
        if self.use_ann:
            try:
                self._maybe_init_ann()
            except ImportError:
                # hnswlib غير مثبت، نرجع للبحث الخطي
                self.use_ann = False
            else:
                if self.ann_index is not None:
                    indices, distances = self.ann_index.knn_query(query_vector, k=min(k, len(ids)))
                    sims = self._convert_ann_distances_to_similarity(np.asarray(distances))

                    results = []
                    for idx, score in zip(indices, sims):
                        results.append(
                            {
                                "id": ids[int(idx)],
                                "score": float(score),
                                "metadata": metas[int(idx)],
                            }
                        )
                    return results

        # -------------------------
        # البحث الخطي (brute-force)
        # -------------------------
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
            "use_ann": self.use_ann,
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
        use_ann = data.get("use_ann", False)
        db = cls(dim=dim, embeddings=embeddings, metric=metric, use_ann=use_ann)
        db.storage = InMemoryStorage.from_dict(data["storage"])
        return db
