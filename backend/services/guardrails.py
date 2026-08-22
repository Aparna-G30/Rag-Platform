"""
Guardrails for the voice-RAG pipeline.

Three checks, applied in order:
  1. Input safety — blocks obviously unsafe/inappropriate queries before they hit retrieval.
  2. Groundedness (relevance gate) — if the best retrieved chunk isn't similar enough
     to the query, we refuse to answer rather than let the LLM improvise.
  3. Output groundedness — after generation, verify the answer isn't citing sources
     that weren't actually retrieved, and isn't suspiciously unsupported by the context.
"""
import re

# --- 1. Input safety -------------------------------------------------------

UNSAFE_PATTERNS = [
    r"\bhow (to|do i) (make|build|synthesi[sz]e)\b.*\b(bomb|explosive|weapon|virus|malware)\b",
    r"\bkill (myself|someone)\b",
    r"\bhack (into|the)\b",
    r"\bchild (sexual|abuse)\b",
]

_compiled_unsafe = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]


def check_input_safety(query: str) -> dict:
    """Returns {'safe': bool, 'reason': str | None}"""
    if not query or not query.strip():
        return {"safe": False, "reason": "Empty query."}
    for pattern in _compiled_unsafe:
        if pattern.search(query):
            return {"safe": False, "reason": "Query matched an unsafe-content pattern."}
    return {"safe": True, "reason": None}


# --- 2. Relevance gate (off-topic detection) -------------------------------

# pgvector cosine_distance: 0 = identical, 2 = opposite. We convert distance -> similarity.
# Tune this threshold against your dataset — start conservative and adjust from eval data.
MIN_RELEVANCE_SIMILARITY = 0.35


def check_relevance(top_chunk_distance: float | None) -> dict:
    """
    top_chunk_distance: the cosine_distance of the *best* retrieved chunk (lower = more similar).
    Returns {'relevant': bool, 'similarity': float | None, 'reason': str | None}
    """
    if top_chunk_distance is None:
        return {"relevant": False, "similarity": None, "reason": "No chunks retrieved."}

    similarity = 1 - top_chunk_distance  # cosine distance -> cosine similarity
    if similarity < MIN_RELEVANCE_SIMILARITY:
        return {
            "relevant": False,
            "similarity": similarity,
            "reason": f"Best match similarity {similarity:.2f} below threshold {MIN_RELEVANCE_SIMILARITY}.",
        }
    return {"relevant": True, "similarity": similarity, "reason": None}


# --- 3. Output groundedness -------------------------------------------------

REFUSAL_PHRASES = ["cannot find this in the document", "not in the sources", "i don't know"]


def check_groundedness(answer: str, chunks: list[dict]) -> dict:
    """
    Lightweight groundedness check (no extra LLM call, to stay latency-friendly):
      - Flags if the model cited a source number that doesn't exist in the retrieved set.
      - Flags if the answer is suspiciously long relative to context (possible fabrication)
        while citing nothing at all.
    Returns {'grounded': bool, 'reason': str | None}
    """
    lower = answer.lower()
    if any(phrase in lower for phrase in REFUSAL_PHRASES):
        # Model itself declined — that's a correctly-working guardrail, not a failure.
        return {"grounded": True, "reason": "Model declined to answer (expected refusal)."}

    cited = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))
    valid_range = set(range(1, len(chunks) + 1))
    bogus_citations = cited - valid_range
    if bogus_citations:
        return {
            "grounded": False,
            "reason": f"Answer cites source(s) {sorted(bogus_citations)} that were never retrieved.",
        }

    if chunks and not cited:
        return {
            "grounded": False,
            "reason": "Answer makes claims without citing any retrieved source.",
        }

    return {"grounded": True, "reason": None}
