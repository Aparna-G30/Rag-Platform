"""
Orchestration layer for the voice RAG pipeline.

This is the "harness" required by the task: instead of a single raw
prompt-in/text-out call, every stage is timed, validated, retried where it
makes sense, and can short-circuit through a guardrail with a structured
refusal rather than pushing bad input further down the pipeline.

Pipeline: audio bytes -> STT -> input safety -> embed -> retrieve
          -> relevance gate -> generate (with retry) -> groundedness check
"""
import time
from dataclasses import dataclass, field
from sqlalchemy.orm import Session

from backend.services.stt import transcribe_audio, STTError
from backend.services.embedder import embed_query
from backend.services.retriever import retrieve_chunks
from backend.services.generator import generate_answer
from backend.services.guardrails import (
    check_input_safety,
    check_relevance,
    check_groundedness,
)


@dataclass
class StageTiming:
    stage: str
    ms: float


@dataclass
class PipelineResult:
    status: str  # "ok" | "refused" | "error"
    answer: str | None
    sources: list[dict] = field(default_factory=list)
    transcript: str | None = None
    refusal_reason: str | None = None
    error: str | None = None
    timings_ms: dict = field(default_factory=dict)
    total_ms: float = 0.0


def _generate_with_retry(query: str, chunks: list[dict], max_attempts: int = 2) -> dict:
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return generate_answer(query, chunks)
        except Exception as e:  # LLM/network hiccup — retry once before giving up
            last_exc = e
            time.sleep(0.3 * (attempt + 1))
    raise RuntimeError(f"Generation failed after {max_attempts} attempts: {last_exc}")


def run_voice_qa_pipeline(
    db: Session,
    audio_bytes: bytes,
    filename: str,
    top_k: int = 5,
    document_id: int | None = None,
    language_code: str = "en-IN",
) -> PipelineResult:
    timings: dict[str, float] = {}
    t_total_start = time.perf_counter()

    # --- Stage 1: Speech-to-text ---
    t0 = time.perf_counter()
    try:
        stt_result = transcribe_audio(audio_bytes, filename=filename, language_code=language_code)
    except STTError as e:
        return PipelineResult(
            status="error",
            answer=None,
            error=f"STT failed: {e}",
            timings_ms={"stt_ms": (time.perf_counter() - t0) * 1000},
            total_ms=(time.perf_counter() - t_total_start) * 1000,
        )
    timings["stt_ms"] = (time.perf_counter() - t0) * 1000
    transcript = stt_result["transcript"]

    # --- Stage 2: Input safety guardrail ---
    t0 = time.perf_counter()
    safety = check_input_safety(transcript)
    timings["input_safety_ms"] = (time.perf_counter() - t0) * 1000
    if not safety["safe"]:
        return PipelineResult(
            status="refused",
            answer=None,
            transcript=transcript,
            refusal_reason=safety["reason"],
            timings_ms=timings,
            total_ms=(time.perf_counter() - t_total_start) * 1000,
        )

    # --- Stage 3: Embed query ---
    t0 = time.perf_counter()
    try:
        query_embedding = embed_query(transcript)
    except Exception as e:
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            status="error", answer=None, transcript=transcript,
            error=f"Embedding failed: {e}", timings_ms=timings,
            total_ms=(time.perf_counter() - t_total_start) * 1000,
        )
    timings["embed_ms"] = (time.perf_counter() - t0) * 1000

    # --- Stage 4: Retrieve ---
    t0 = time.perf_counter()
    chunks = retrieve_chunks(db, query_embedding, top_k=top_k, document_id=document_id)
    timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

    # --- Stage 5: Relevance guardrail (off-topic detection) ---
    t0 = time.perf_counter()
    best_distance = chunks[0]["distance"] if chunks else None
    relevance = check_relevance(best_distance)
    timings["relevance_check_ms"] = (time.perf_counter() - t0) * 1000
    if not relevance["relevant"]:
        return PipelineResult(
            status="refused",
            answer=None,
            transcript=transcript,
            sources=chunks,
            refusal_reason=relevance["reason"] or "Query appears off-topic for this dataset.",
            timings_ms=timings,
            total_ms=(time.perf_counter() - t_total_start) * 1000,
        )

    # --- Stage 6: Generate (with retry) ---
    t0 = time.perf_counter()
    try:
        gen_result = _generate_with_retry(transcript, chunks)
    except Exception as e:
        timings["generation_ms"] = (time.perf_counter() - t0) * 1000
        return PipelineResult(
            status="error", answer=None, transcript=transcript, sources=chunks,
            error=str(e), timings_ms=timings,
            total_ms=(time.perf_counter() - t_total_start) * 1000,
        )
    timings["generation_ms"] = (time.perf_counter() - t0) * 1000
    answer = gen_result["answer"]

    # --- Stage 7: Output groundedness guardrail ---
    t0 = time.perf_counter()
    groundedness = check_groundedness(answer, chunks)
    timings["groundedness_check_ms"] = (time.perf_counter() - t0) * 1000
    if not groundedness["grounded"]:
        return PipelineResult(
            status="refused",
            answer=None,
            transcript=transcript,
            sources=chunks,
            refusal_reason=f"Answer failed groundedness check: {groundedness['reason']}",
            timings_ms=timings,
            total_ms=(time.perf_counter() - t_total_start) * 1000,
        )

    return PipelineResult(
        status="ok",
        answer=answer,
        transcript=transcript,
        sources=chunks,
        timings_ms=timings,
        total_ms=(time.perf_counter() - t_total_start) * 1000,
    )
