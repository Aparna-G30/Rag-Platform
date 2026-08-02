from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from backend.db import get_db
from backend.models.document import Document, Chunk
from backend.services.extractor import extract_text
from backend.services.chunker import chunk_pages
from backend.services.embedder import embed_chunks
import shutil, os

router = APIRouter(prefix="/documents", tags=["documents"])
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Save file
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2. Save document record
    doc = Document(filename=file.filename)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # 3. Extract → chunk → embed
    pages = extract_text(file_path)
    chunks = chunk_pages(pages)
    embeddings = embed_chunks([c["content"] for c in chunks])

    # 4. Store chunks + vectors
    for chunk_data, embedding in zip(chunks, embeddings):
        chunk = Chunk(
            document_id=doc.id,
            content=chunk_data["content"],
            page_number=chunk_data["page_number"],
            embedding=embedding
        )
        db.add(chunk)

    doc.total_chunks = len(chunks)
    db.commit()

    return {
        "id": doc.id,
        "filename": doc.filename,
        "chunks_created": len(chunks),
        "status": "ingested"
    }

@router.get("/")
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [{"id": d.id, "filename": d.filename,
             "chunks": d.total_chunks, "created_at": str(d.created_at)}
            for d in docs]

@router.delete("/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    db.query(Chunk).filter(Chunk.document_id == doc_id).delete()
    db.query(Document).filter(Document.id == doc_id).delete()
    db.commit()
    return {"status": "deleted"}