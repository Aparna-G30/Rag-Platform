from fastapi import FastAPI

app = FastAPI(title="RAG Platform")

@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi.middleware.cors import CORSMiddleware
from backend.routes.documents import router as doc_router

app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"], allow_headers=["*"])
app.include_router(doc_router)