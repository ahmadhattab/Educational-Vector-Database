import numpy as np

def _safe_normalize(vectors: np.ndarray) -> np.ndarray:
    eps = 1e-12
    if vectors.ndim == 1:
        return vectors / (np.linalg.norm(vectors) + eps)
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + eps)

def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    q = _safe_normalize(query)
    v = _safe_normalize(vectors)
    return v @ q
