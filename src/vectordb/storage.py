import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np


class InMemoryStorage:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: List[np.ndarray] = []
        self._metadatas: List[Dict] = []
        self._ids: List[str] = []

    def add(self, vector: np.ndarray, metadata: Optional[Dict] = None) -> str:
        if vector.shape[-1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {vector.shape[-1]}")
        _id = str(uuid.uuid4())
        self._vectors.append(vector.astype(float))
        self._metadatas.append(metadata or {})
        self._ids.append(_id)
        return _id

    def get_all(self) -> Tuple[np.ndarray, List[Dict], List[str]]:
        if not self._vectors:
            return np.empty((0, self.dim)), [], []
        return np.vstack(self._vectors), self._metadatas, self._ids

    # -------- v2: دعم التحويل لقاموس والتحميل منه --------

    def to_dict(self) -> Dict:
        if self._vectors:
            vectors = np.vstack(self._vectors)
        else:
            vectors = np.empty((0, self.dim))
        return {
            "dim": self.dim,
            "vectors": vectors,
            "metadatas": self._metadatas,
            "ids": self._ids,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "InMemoryStorage":
        dim = data["dim"]
        storage = cls(dim=dim)
        vectors = data.get("vectors", np.empty((0, dim)))
        metadatas = data.get("metadatas", [])
        ids = data.get("ids", [])

        storage._vectors = [v.astype(float) for v in vectors]  # نفصل الصفوف
        storage._metadatas = list(metadatas)
        storage._ids = list(ids)

        return storage
