"""
HNSW-based Approximate Nearest Neighbor (ANN) index.

الفكرة: البحث الخطي (brute-force) ممتاز للبيانات الصغيرة،
لكن لما يزيد عدد المتجهات يصير أبطأ بكثير.
هنا نستخدم HNSW (Hierarchical Navigable Small World graph)
لبناء فهرس يسمح لنا نجيب أقرب المتجهات بسرعة تقريبية.

هذا الكود تعليمي ومبسّط:
- يبني الفهرس من مصفوفة متجهات كاملة.
- يدعم إضافة متجهات جديدة لاحقاً (add_items).
- يدير كل شيء داخل كلاس صغير سهل القراءة.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


class HNSWIndex:
    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
    ):
        """
        dim: أبعاد المتجهات.
        metric: cosine أو euclidean أو dot (ip).
        ef_construction / M: بارامترات تحكم جودة وسرعة البناء.
        ef_search: مدى توسّع البحث (أكبر = أدق لكن أبطأ قليلاً).
        """
        try:
            import hnswlib  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "hnswlib غير مثبت. أضِفه إلى متطلباتك: pip install hnswlib"
            ) from exc

        self.hnswlib = hnswlib
        self.dim = dim
        self.metric = metric
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        self._index = None

        # تحويل metric إلى space مفهوم بالنسبة لـ hnswlib.
        self.space = {
            "cosine": "cosine",
            "euclidean": "l2",
            "dot": "ip",  # inner product
        }.get(metric, "cosine")

    def _init_index(self, max_elements: int):
        self._index = self.hnswlib.Index(space=self.space, dim=self.dim)
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self._index.set_ef(self.ef_search)

    def build(self, vectors: np.ndarray):
        """
        بناء الفهرس من الصفر باستخدام جميع المتجهات الحالية.
        """
        if vectors.shape[0] == 0:
            return

        self._init_index(max_elements=vectors.shape[0])
        ids = list(range(vectors.shape[0]))
        self._index.add_items(vectors, ids)

    def add_items(self, vectors: np.ndarray, ids: List[int]):
        """
        إضافة عناصر جديدة بشكل تدريجي.
        لو الفهرس غير موجود سنبنيه أولاً.
        """
        if self._index is None:
            max_elements = max(ids) + 1 if ids else vectors.shape[0]
            self._init_index(max_elements=max_elements)
        else:
            required_size = max(ids) + 1 if ids else vectors.shape[0]
            if required_size > self._index.get_max_elements():
                self._index.resize_index(required_size)

        self._index.add_items(vectors, ids)

    def knn_query(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        استعلام أقرب الجيران.
        يرجع (indices, distances) مثل hnswlib.
        """
        if self._index is None:
            return np.array([]), np.array([])

        labels, distances = self._index.knn_query(query_vector, k=k)
        return labels[0], distances[0]
