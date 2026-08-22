from sqlalchemy import Column,Integer,String,DateTime,Text
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from backend.db import Base

class Document(Base):
    __tablename__="documents"
    id=Column(Integer,primary_key=True)
    filename=Column(String,nullable=False)
    total_chunks=Column(Integer,default=0)
    created_at=Column(DateTime,server_default=func.now())

class Chunk(Base):
    __tablename__="chunks"
    id=Column(Integer,primary_key=True)
    document_id=Column(Integer,nullable=False)
    content=Column(Text,nullable=False)
    page_number=Column(Integer)
    embedding=Column(Vector(1024)) # 1024 dims for Cohere embed-v3
    created_at=Column(DateTime,server_default=func.now())
    # metadata for dataset-sourced chunks (e.g. MSMARCO-XI ingestion)
    chunk_strategy=Column(String, nullable=True)   # "fixed" | "sentence" | "semantic"
    language=Column(String, nullable=True)          # e.g. "hi", "en"
    source_query_id=Column(String, nullable=True)   # groups chunks back to their source query/passage set