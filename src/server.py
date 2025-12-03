from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from vectordb import VectorDB, DummyEmbeddings
from vectordb.embeddings import HFSentenceTransformerEmbeddings
from vectordb.rag import build_rag_index, answer_query


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


def load_or_init_db(use_ann: bool = True):
    embeddings = build_embeddings()
    if DB_SAVE_PATH.exists():
        db_loaded = VectorDB.load(str(DB_SAVE_PATH), embeddings=embeddings)
        # نسمح بتفعيل/إلغاء ANN بعد التحميل أيضاً
        db_loaded.use_ann = use_ann
        return db_loaded

    dim = embeddings.embed("test").shape[0]
    return VectorDB(dim=dim, embeddings=embeddings, metric="cosine", use_ann=use_ann)


def persist_db(db: VectorDB):
    DB_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    db.save(str(DB_SAVE_PATH))


db = load_or_init_db()

app = FastAPI(title="Educational Vector DB API", version="0.3.0")


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


class RagIndexRequest(BaseModel):
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    chunk_size: int = 300


class RagAnswerRequest(BaseModel):
    query: str
    k: int = 3


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


@app.get("/items")
def list_items():
    """
    إرجاع العناصر المخزنة لعرضها في لوحة التحكم.
    """
    vectors, metas, ids = db.storage.get_all()
    items = []
    for idx, item_id in enumerate(ids):
        items.append(
            {
                "id": item_id,
                "metadata": metas[idx],
            }
        )
    return items


@app.post("/rag/index")
def rag_index(req: RagIndexRequest):
    build_rag_index(
        db,
        texts=req.texts,
        metadatas=req.metadatas,
        chunk_size=req.chunk_size,
    )
    persist_db(db)
    return {"status": "ok", "count": len(req.texts)}


@app.post("/rag/answer")
def rag_answer(req: RagAnswerRequest):
    return answer_query(db, req.query, top_k=req.k)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """
    صفحة HTML بسيطة مع واجهة عربية خفيفة لعرض العناصر والبحث.
    """
    return """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8" />
  <title>لوحة التحكم - قاعدة المتجهات التعليمية</title>
  <style>
    body { font-family: 'Cairo', 'Segoe UI', sans-serif; margin: 0; background: #f4f6fb; color: #1f2a44; }
    header { background: linear-gradient(120deg, #3a7bd5, #00d2ff); color: white; padding: 24px; }
    h1 { margin: 0 0 6px 0; }
    p.sub { margin: 0; opacity: 0.9; }
    main { padding: 20px; max-width: 1000px; margin: 0 auto; }
    section { background: white; border-radius: 12px; padding: 16px; margin-bottom: 16px; box-shadow: 0 8px 20px rgba(0,0,0,0.05); }
    label { display: block; font-weight: 600; margin-bottom: 6px; }
    input, textarea, button { width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #d6d9e0; box-sizing: border-box; }
    button { background: #3a7bd5; color: white; border: none; cursor: pointer; font-weight: 700; transition: transform 0.1s ease, box-shadow 0.1s ease; }
    button:hover { transform: translateY(-1px); box-shadow: 0 6px 12px rgba(58,123,213,0.25); }
    .flex { display: flex; gap: 12px; }
    .flex > div { flex: 1; }
    .item { padding: 10px; border-bottom: 1px solid #eef1f7; }
    .item:last-child { border-bottom: none; }
    .meta { font-size: 12px; color: #697386; }
    .score { color: #0c8e63; font-weight: 700; }
    .badge { display: inline-block; background: #eef4ff; color: #3a7bd5; padding: 2px 8px; border-radius: 6px; font-size: 12px; margin-left: 6px; }
    .results { margin-top: 10px; }
  </style>
</head>
<body>
  <header>
    <h1>لوحة التحكم</h1>
    <p class="sub">استعراض العناصر والبحث السريع (بحث تقليدي أو ANN بحسب الإعدادات).</p>
  </header>
  <main>
    <section>
      <div class="flex">
        <div>
          <label for="query">استعلام البحث</label>
          <input id="query" placeholder="مثال: تعلم الآلة" />
        </div>
        <div>
          <label for="k">عدد النتائج (k)</label>
          <input id="k" type="number" value="5" min="1" />
        </div>
      </div>
      <button onclick="runSearch()">بحث الآن</button>
      <div id="results" class="results"></div>
    </section>

    <section>
      <h3>النصوص المخزنة</h3>
      <div id="items"></div>
    </section>
  </main>

  <script>
    async function fetchJSON(url, options) {
      const res = await fetch(url, options);
      return await res.json();
    }

    async function loadItems() {
      const data = await fetchJSON('/items');
      const container = document.getElementById('items');
      if (!data.length) {
        container.innerHTML = '<p>لا توجد عناصر بعد.</p>';
        return;
      }
      container.innerHTML = data.map(item => {
        const text = item.metadata?.text || '';
        const meta = {...item.metadata};
        delete meta.text;
        return `
          <div class="item">
            <div><strong>نص:</strong> ${text || '---'}</div>
            <div class="meta">id: ${item.id}</div>
            <div class="meta">metadata: ${JSON.stringify(meta)}</div>
          </div>
        `;
      }).join('');
    }

    async function runSearch() {
      const query = document.getElementById('query').value;
      const k = parseInt(document.getElementById('k').value || '5', 10);
      const data = await fetchJSON('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, k })
      });
      const container = document.getElementById('results');
      if (!data.length) {
        container.innerHTML = '<p>لا نتائج بعد.</p>';
        return;
      }
      container.innerHTML = data.map(r => {
        const text = r.metadata?.text || '';
        const meta = {...r.metadata};
        delete meta.text;
        return `
          <div class="item">
            <div><span class="badge">score</span> <span class="score">${r.score.toFixed(4)}</span></div>
            <div><strong>نص:</strong> ${text || '---'}</div>
            <div class="meta">metadata: ${JSON.stringify(meta)}</div>
          </div>
        `;
      }).join('');
    }

    loadItems();
  </script>
</body>
</html>
    """
