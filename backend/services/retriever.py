"""
Thin retrieval wrapper around the existing Chunk/pgvector query.
Separated out so both the text-search route and the voice harness can share it,
and so we can expose cosine_distance for the relevance guardrail (the original
/search/ route only returned chunk content, not the distance).
"""
from sqlalchemy.orm import Session
from sqlalchemy import select
from backend.models.document import Chunk


def retrieve_chunks(db: Session, query_embedding: list[float], top_k: int = 5,
                     document_id: int | None = None) -> list[dict]:
    """
    Returns chunks ordered by similarity, each annotated with its cosine_distance
    (0 = identical, 2 = opposite; convert to similarity as 1 - distance).
    """
    distance_expr = Chunk.embedding.cosine_distance(query_embedding)
    stmt = select(Chunk, distance_expr.label("distance")).order_by(distance_expr)

    if document_id:
        stmt = stmt.where(Chunk.document_id == document_id)

    stmt = stmt.limit(top_k)
    rows = db.execute(stmt).all()

    return [
        {
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "page_number": chunk.page_number,
            "content": chunk.content,
            "distance": float(distance),
        }
        for chunk, distance in rows
    ]
