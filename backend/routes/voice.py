from fastapi import APIRouter, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
from backend.db import get_db
from backend.services.harness import run_voice_qa_pipeline

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/qa")
async def voice_qa(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    document_id: int | None = Form(None),
    language_code: str = Form("en-IN"),
    db: Session = Depends(get_db),
):
    """
    End-to-end voice RAG: audio upload -> transcript -> retrieval -> grounded answer.
    Returns per-stage latency in milliseconds and refuses to answer when the
    query is unsafe, off-topic, or the generated answer isn't grounded in context.
    """
    audio_bytes = await file.read()

    result = run_voice_qa_pipeline(
        db=db,
        audio_bytes=audio_bytes,
        filename=file.filename or "audio.wav",
        top_k=top_k,
        document_id=document_id,
        language_code=language_code,
    )

    response = {
        "status": result.status,
        "transcript": result.transcript,
        "answer": result.answer,
        "sources": result.sources,
        "timings_ms": {k: round(v, 2) for k, v in result.timings_ms.items()},
        "total_ms": round(result.total_ms, 2),
    }
    if result.status == "refused":
        response["refusal_reason"] = result.refusal_reason
    if result.status == "error":
        response["error"] = result.error

    return response
