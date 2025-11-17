from abc import ABC, abstractmethod
import numpy as np

class Embeddings(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        ...

class DummyEmbeddings(Embeddings):
        def __init__(self, dim: int):
            self.dim = dim

        def embed(self, text: str) -> np.ndarray:
            np.random.seed(len(text))
            return np.random.rand(self.dim)
