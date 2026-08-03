from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from backend.db import get_db
from backend.models.document import Chunk
from backend.services.embedder import embed_query

from backend.services.generator import generate_answer

class QARequest(BaseModel):
    query: str
    document_id: int | None = None
    top_k: int = 5

router = APIRouter(prefix="/search", tags=["search"])

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    document_id: int | None = None

@router.post("/")
def semantic_search(req: SearchRequest, db: Session = Depends(get_db)):
    query_embedding = embed_query(req.query)

    q = db.query(Chunk).order_by(
        Chunk.embedding.cosine_distance(query_embedding)
    )
    if req.document_id:
        q = q.filter(Chunk.document_id == req.document_id)

    results = q.limit(req.top_k).all()
    return [{"chunk_id": r.id, "document_id": r.document_id,
             "page_number": r.page_number, "content": r.content}
            for r in results]

@router.post("/qa")
def ask_question(req: QARequest, db: Session = Depends(get_db)):
    search_req = SearchRequest(query=req.query, top_k=req.top_k,
                               document_id=req.document_id)
    chunks = semantic_search(search_req, db)

    if not chunks:
        return {"answer": "No relevant content found.", "sources": []}

    result = generate_answer(req.query, chunks)
    return {"answer": result["answer"], "sources": chunks,
            "sources_used": result["sources_used"]}