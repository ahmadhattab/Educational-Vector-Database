import numpy as np

from vectordb import VectorDB, DummyEmbeddings


def test_add_and_search_text():
    dim = 8
    embeddings = DummyEmbeddings(dim=dim)
    db = VectorDB(dim=dim, embeddings=embeddings)

    # نضيف نصوص مختلفة
    t1 = "I love machine learning"
    t2 = "Football is my favourite sport"

    id1 = db.add_text(t1, metadata={"source": "test"})
    id2 = db.add_text(t2, metadata={"source": "test"})

    assert isinstance(id1, str)
    assert isinstance(id2, str)
    assert id1 != id2

    # نبحث عن شيء قريب من machine learning
    results = db.search_text("deep learning", k=1)

    # مهم يرجع نتيجة واحدة على الأقل
    assert len(results) == 1

    top = results[0]
    assert "id" in top
    assert "score" in top
    assert "metadata" in top

    # نتأكد إن الـ metadata فيها النص الأصلي
    assert "text" in top["metadata"]
    assert isinstance(top["metadata"]["text"], str)
