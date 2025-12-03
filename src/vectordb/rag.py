"""
RAG (Retrieval-Augmented Generation) مبسّط.

الهدف هنا تعليمي:
- تقسيم النصوص الطويلة إلى قطع صغيرة يسهل استرجاعها.
- تخزينها في قاعدة المتجهات.
- عند الاستعلام نعيد أفضل القطع + نجمعها في نص واحد.
- خانة "answer" مجرد placeholder يوضح مكان استدعاء LLM لاحقاً.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .api import VectorDB


def _chunk_text(text: str, chunk_size: int) -> List[str]:
    """
    تقسيم بسيط للنص بناءً على عدد الحروف.
    يمكن استبداله لاحقاً بتقسيم ذكي (حسب الجمل/الفقرات).
    """
    if not text:
        return [""]
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def build_rag_index(
    db: VectorDB,
    texts: List[str],
    metadatas: Optional[List[Dict]] = None,
    chunk_size: int = 300,
) -> None:
    """
    يقسم النصوص إلى chunks ويضيفها إلى القاعدة.

    metadatas: قائمة موازية للنصوص (يمكن أن تكون None).
    chunk_size: طول القطعة التقريبي بالحروف.
    """
    metadatas = metadatas or [{} for _ in texts]

    for idx, text in enumerate(texts):
        base_meta = metadatas[idx] if metadatas and idx < len(metadatas) else {}
        chunks = _chunk_text(text, chunk_size=chunk_size)
        for chunk_id, chunk in enumerate(chunks):
            metadata = base_meta.copy()
            metadata.update(
                {
                    "doc_index": idx,
                    "chunk_index": chunk_id,
                }
            )
            db.add_text(chunk, metadata=metadata)


def index_documents(
    db: VectorDB,
    texts: List[str],
    metadata_list: Optional[List[Dict]] = None,
    chunk_size: int = 300,
) -> None:
    """
    اسم بديل لنفس الوظيفة لإرضاء من يفضل "index_documents".
    """
    return build_rag_index(db, texts, metadatas=metadata_list, chunk_size=chunk_size)


def answer_query(db: VectorDB, query: str, top_k: int = 3) -> Dict:
    """
    يسترجع أفضل القطع ويجمعها كنص واحد.
    answer هنا placeholder فقط، وفي نظام حقيقي ستستدعي LLM.
    """
    results = db.search_text(query, k=top_k)

    chunks = []
    for r in results:
        meta = r.get("metadata", {})
        chunks.append(
            {
                "id": r["id"],
                "score": r["score"],
                "text": meta.get("text", ""),
                "metadata": meta,
            }
        )

    combined_context = "\n---\n".join(c["text"] for c in chunks)
    answer = (
        "This is a placeholder answer. Relevant context:\n"
        f"{combined_context}"
    )

    return {
        "query": query,
        "chunks": chunks,
        "combined_context": combined_context,
        "answer": answer,
    }
