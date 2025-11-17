import uuid
import numpy as np
from typing import List, Dict, Tuple

class InMemoryStorage:
    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: List[np.ndarray] = []
        self._metadatas: List[Dict] = []
        self._ids: List[str] = []

    def add(self, vector: np.ndarray, metadata: Dict | None = None) -> str:
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
