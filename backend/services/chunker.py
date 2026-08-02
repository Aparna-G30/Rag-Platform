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