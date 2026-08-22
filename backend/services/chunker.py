def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.
    chunk_size: target words per chunk
    overlap: words shared between consecutive chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # slide with overlap
    return chunks

def chunk_pages(pages: list[dict]) -> list[dict]:
    all_chunks = []
    for page in pages:
        for chunk in chunk_text(page["text"]):
            all_chunks.append({
                "content": chunk,
                "page_number": page["page_number"]
            })
    return all_chunks


# --- Strategy 2: sentence-aware fixed-size (keeps sentences whole, doesn't cut mid-sentence) ---

import re


def sentence_chunk_text(text: str, max_words: int = 150, overlap_sentences: int = 1) -> list[str]:
    """
    Splits on sentence boundaries and groups sentences up to max_words per chunk.
    Avoids the fixed-size strategy's problem of cutting a sentence in half.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    chunks = []
    current, current_words = [], 0

    for sent in sentences:
        sent_words = len(sent.split())
        if current and current_words + sent_words > max_words:
            chunks.append(" ".join(current))
            # keep the last N sentences for overlap/continuity
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_words = sum(len(s.split()) for s in current)
        current.append(sent)
        current_words += sent_words

    if current:
        chunks.append(" ".join(current))
    return chunks


# --- Strategy 3: semantic chunking (splits where meaning shifts, using embedding similarity) ---

def semantic_chunk_text(text: str, embed_fn, similarity_threshold: float = 0.5,
                         min_sentences: int = 2) -> list[str]:
    """
    Splits text into sentences, embeds each, and starts a new chunk wherever
    consecutive sentences fall below `similarity_threshold` (i.e. meaning shifts).
    embed_fn: a function(list[str]) -> list[list[float]], e.g. embedder.embed_chunks.

    This is the "does real thought go into splitting" strategy — content that
    stays semantically coherent gets grouped together regardless of raw length.
    """
    import numpy as np

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    if len(sentences) <= min_sentences:
        return [text.strip()] if text.strip() else []

    embeddings = embed_fn(sentences)
    chunks, current = [], [sentences[0]]

    for i in range(1, len(sentences)):
        a, b = np.array(embeddings[i - 1]), np.array(embeddings[i])
        cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        if cos_sim < similarity_threshold and len(current) >= min_sentences:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current:
        chunks.append(" ".join(current))
    return chunks


# --- Strategy 4: metadata-aware chunking (attaches source metadata to every chunk) ---

def metadata_aware_chunk(text: str, metadata: dict, strategy: str = "sentence", **kwargs) -> list[dict]:
    """
    Wraps any of the strategies above and attaches metadata (query_id, language,
    passage_index, is_selected, etc. — whatever the caller passes) to every
    resulting chunk. This is what lets retrieval later filter/boost by source
    structure instead of treating every chunk as an anonymous blob of text.
    """
    if strategy == "fixed":
        raw_chunks = chunk_text(text, **kwargs)
    elif strategy == "sentence":
        raw_chunks = sentence_chunk_text(text, **kwargs)
    elif strategy == "semantic":
        raw_chunks = semantic_chunk_text(text, **kwargs)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return [
        {"content": c, "chunk_strategy": strategy, **metadata}
        for c in raw_chunks
    ]