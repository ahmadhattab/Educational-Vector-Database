import pathlib
import sys

import numpy as np
import pytest

from vectordb import VectorDB
from vectordb.index import similarity_scores


def test_save_and_load_roundtrip(tmp_path):
    """
    نختبر حفظ وتحميل قاعدة البيانات باستخدام متجهات بسيطة معروفة.
    """
    db_path = tmp_path / "vdb.pkl"

    dim = 2
    db = VectorDB(dim=dim, embeddings=None, metric="cosine")

    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    db.add_vector(v1, {"label": "x"})
    db.add_vector(v2, {"label": "y"})

    # حفظ على القرص
    db.save(str(db_path))
    assert db_path.exists()

    # تحميل من القرص
    db2 = VectorDB.load(str(db_path), embeddings=None)

    # تأكد أن الإعدادات الأساسية محفوظة
    assert db2.dim == db.dim
    assert db2.metric == db.metric

    # تأكد أن المتجهات موجودة بعد التحميل
    vectors, metadatas, ids = db2.storage.get_all()
    assert vectors.shape == (2, dim)
    assert len(metadatas) == 2
    assert len(ids) == 2

    # تأكد أن البحث ما زال يعطي نتيجة منطقية
    query = np.array([1.0, 0.0])
    results = db2.search_vector(query, k=1)
    assert len(results) == 1
    top = results[0]
    # أقرب متجه لـ [1,0] هو الأول
    assert top["metadata"]["label"] == "x"


def test_similarity_metrics_select_same_best_vector():
    """
    نختبر أن كل metric (cosine / dot / euclidean-as-similarity)
    تعطي أعلى تشابه لنفس المتجه في مثال بسيط.
    """
    query = np.array([1.0, 0.0])
    vectors = np.array([
        [1.0, 0.0],  # الأقرب
        [0.0, 1.0],
    ])

    for metric in ["cosine", "dot", "euclidean"]:
        scores = similarity_scores(query, vectors, metric=metric)
        # أعلى قيمة لازم تكون لأول متجه
        assert scores.shape == (2,)
        assert int(np.argmax(scores)) == 0

    # metric غير معروف لازم يرفع خطأ
    with pytest.raises(ValueError):
        similarity_scores(query, vectors, metric="unknown-metric")


def import_server_with_dummy_embeddings(monkeypatch):
    """
    نستورد server بطريقة تتحكم بالـ Embeddings حتى لا نحاول
    تحميل sentence-transformers فعلياً أثناء الاختبارات.
    """
    # نضمن أن src/ موجود على المسار
    root = pathlib.Path(__file__).resolve().parents[1]
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    import vectordb.embeddings as emb_mod

    class FakeHF(emb_mod.Embeddings):
        def __init__(self, *args, **kwargs):
            # نجبر build_embeddings إنه يعمل fallback على DummyEmbeddings
            raise ModuleNotFoundError("force DummyEmbeddings in tests")

        def embed(self, text: str):
            raise NotImplementedError

    # نبدّل HFSentenceTransformerEmbeddings قبل استيراد server
    monkeypatch.setattr(emb_mod, "HFSentenceTransformerEmbeddings", FakeHF)

    import importlib
    server = importlib.import_module("server")
    return server


def test_api_helpers_save_and_reload(tmp_path, monkeypatch):
    """
    نختبر load_or_init_db + persist_db في server.py
    بدون لمس موديلات حقيقية ثقيلة.
    """
    server = import_server_with_dummy_embeddings(monkeypatch)

    # نوجه مسار الحفظ إلى مجلد مؤقت للاختبار
    server.DB_SAVE_PATH = tmp_path / "vectordb_api.pkl"

    # ننشئ DB جديدة باستخدام load_or_init_db (ما فيه ملف حالياً)
    db = server.load_or_init_db()
    assert isinstance(db, VectorDB)

    # نضيف بيانات بسيطة
    db.add_text("hello world", metadata={"source": "test"})
    db.add_text("another item", metadata={"source": "test"})

    # نحفظ باستخدام helper
    server.persist_db(db)
    assert server.DB_SAVE_PATH.exists()

    # نحمل باستخدام API الأساسية لـ VectorDB
    db2 = VectorDB.load(str(server.DB_SAVE_PATH), embeddings=db.embeddings)

    vectors1, metas1, ids1 = db.storage.get_all()
    vectors2, metas2, ids2 = db2.storage.get_all()

    # نفس عدد المتجهات
    assert vectors2.shape == vectors1.shape
    assert len(metas2) == len(metas1) == 2
    assert len(ids2) == len(ids1) == 2

    # نتحقق أن البحث ما زال يعمل بعد التحميل
    results = db2.search_text("hello", k=1)
    assert len(results) == 1
    assert "metadata" in results[0]
    assert results[0]["metadata"]["source"] == "test"
