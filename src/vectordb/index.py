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


def dot_product(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    return vectors @ query


def euclidean_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    diff = vectors - query.reshape(1, -1)
    return np.sqrt(np.sum(diff ** 2, axis=1))


def similarity_scores(
    query: np.ndarray,
    vectors: np.ndarray,
    metric: str = "cosine"
) -> np.ndarray:
    metric = metric.lower()
    if metric == "cosine":
        return cosine_similarity(query, vectors)
    elif metric == "dot":
        return dot_product(query, vectors)
    elif metric == "euclidean":
        d = euclidean_distance(query, vectors)
        return 1.0 / (1.0 + d)
    else:
        raise ValueError(f"Unknown metric: {metric}")
