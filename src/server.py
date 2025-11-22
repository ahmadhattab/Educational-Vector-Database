from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI
from pydantic import BaseModel

from vectordb import VectorDB, DummyEmbeddings
from vectordb.embeddings import HFSentenceTransformerEmbeddings


DB_SAVE_PATH = Path("data/vectordb.pkl")


def build_embeddings():
    # اختر نوع الـ Embeddings:
    # 1) تعليمي: Dummy
    # 2) حقيقي: HFSentenceTransformerEmbeddings
    #
    # مثال تعليمي (أسرع، ما بده تنزيل موديل):
    # return DummyEmbeddings(dim=16)

    # مثال حقيقي باستخدام sentence-transformers:
    try:
        return HFSentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    except ModuleNotFoundError:
        # Fallback for environments بدون sentence-transformers
        print("sentence-transformers not installed; falling back to DummyEmbeddings.")
        return DummyEmbeddings(dim=16)


def load_or_init_db():
    embeddings = build_embeddings()
    if DB_SAVE_PATH.exists():
        return VectorDB.load(str(DB_SAVE_PATH), embeddings=embeddings)

    dim = embeddings.embed("test").shape[0]
    return VectorDB(dim=dim, embeddings=embeddings, metric="cosine")


def persist_db(db: VectorDB):
    DB_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    db.save(str(DB_SAVE_PATH))


db = load_or_init_db()

app = FastAPI(title="Educational Vector DB API", version="0.2.0")


class AddTextRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None


class AddTextResponse(BaseModel):
    id: str


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]


@app.post("/add_text", response_model=AddTextResponse)
def add_text(req: AddTextRequest):
    _id = db.add_text(req.text, metadata=req.metadata)
    persist_db(db)
    return AddTextResponse(id=_id)


@app.post("/search", response_model=List[SearchResult])
def search(req: SearchRequest):
    results = db.search_text(req.query, k=req.k)
    return [
        SearchResult(
            id=r["id"],
            score=r["score"],
            metadata=r["metadata"],
        )
        for r in results
    ]
