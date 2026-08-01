from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from backend.db import get_db
from backend.models.document import Document
import shutil, os

router = APIRouter(prefix="/documents", tags=["documents"])
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    file_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = Document(filename=file.filename)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    return {"id": doc.id, "filename": doc.filename, "status": "uploaded"}