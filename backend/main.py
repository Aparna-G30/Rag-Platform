from fastapi import FastAPI


app = FastAPI(title="RAG Platform")

@app.get("/health")
def health():
    return {"status": "ok"}

from fastapi.middleware.cors import CORSMiddleware
from backend.routes.documents import router as doc_router

app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"], allow_headers=["*"])
app.include_router(doc_router)

from backend.routes.search import router as search_router
app.include_router(search_router)

from backend.routes.voice import router as voice_router
app.include_router(voice_router)