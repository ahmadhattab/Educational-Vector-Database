import numpy as np

from vectordb import VectorDB, DummyEmbeddings
from vectordb.rag import build_rag_index, answer_query


def test_save_load_with_ann_disabled(tmp_path):
    db_path = tmp_path / "ann_disabled.pkl"
    dim = 3
    db = VectorDB(dim=dim, embeddings=None, metric="cosine", use_ann=False)

    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    db.add_vector(v1, {"label": "a"})
    db.add_vector(v2, {"label": "b"})

    db.save(db_path)
    loaded = VectorDB.load(db_path, embeddings=None)

    assert loaded.dim == db.dim
    assert loaded.metric == db.metric
    assert loaded.use_ann is False

    res = loaded.search_vector(np.array([1.0, 0.0, 0.0]), k=1)
    assert len(res) == 1
    assert res[0]["metadata"]["label"] == "a"


def test_ann_matches_bruteforce_top_result():
    dim = 2
    db_ann = VectorDB(dim=dim, embeddings=None, metric="cosine", use_ann=True)

    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    db_ann.add_vector(v1, {"label": "x"})
    db_ann.add_vector(v2, {"label": "y"})

    query = np.array([0.9, 0.1])
    ann_result = db_ann.search_vector(query, k=1)[0]

    # brute-force reference
    vectors = np.vstack([v1, v2])
    scores = np.array([np.dot(query, v1), np.dot(query, v2)])
    brute_best = int(np.argmax(scores))

    assert ann_result["metadata"]["label"] == ["x", "y"][brute_best]


def test_rag_answer_structure():
    dim = 8
    db = VectorDB(dim=dim, embeddings=DummyEmbeddings(dim=dim))

    texts = [
        "This is a tiny document about AI.",
        "Another note talks about deep learning and vectors.",
    ]
    build_rag_index(db, texts=texts, metadatas=[{"source": "doc1"}, {"source": "doc2"}], chunk_size=50)

    result = answer_query(db, query="deep learning", top_k=2)

    for key in ["query", "chunks", "combined_context", "answer"]:
        assert key in result

    assert len(result["chunks"]) > 0
    assert "deep" in result["combined_context"].lower() or "learning" in result["combined_context"].lower()
