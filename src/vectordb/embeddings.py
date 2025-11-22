from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Embeddings(ABC):
    """
    واجهة مجردة لأي موديل يحول نص إلى متجه.
    """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        ...


class DummyEmbeddings(Embeddings):
    """
    Embeddings تعليمية، بتستخدم random vector ثابت بناء على طول النص.
    """

    def __init__(self, dim: int):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        np.random.seed(len(text))
        return np.random.rand(self.dim)


class HFSentenceTransformerEmbeddings(Embeddings):
    """
    Embeddings حقيقية باستخدام sentence-transformers (HuggingFace)
    مثال على موديل:
    - sentence-transformers/all-MiniLM-L6-v2
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # import داخل الكلاس
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        # الموديل برجع list/np.array، نأحذ أول عنصر
        vec = self.model.encode([text])[0]
        return np.asarray(vec, dtype=float)
