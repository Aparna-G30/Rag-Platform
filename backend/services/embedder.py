import cohere, os, time
from dotenv import load_dotenv
from cohere.errors import TooManyRequestsError

load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

def embed_chunks(texts: list[str]) -> list[list[float]]:
    all_embeddings = []
    batch_size = 32  # smaller batches = smaller bursts
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for attempt in range(6):
            try:
                response = co.embed(
                    texts=batch,
                    model="embed-english-v3.0",
                    input_type="search_document"
                )
                all_embeddings.extend(response.embeddings)
                break
            except TooManyRequestsError:
                wait = 20 * (attempt + 1)
                print(f"Rate limited on batch {i}, waiting {wait}s (attempt {attempt+1}/6)...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to embed batch starting at index {i} after 6 attempts")
        time.sleep(8)  # pacing between batches
    return all_embeddings

def embed_query(query: str) -> list[float]:
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    return response.embeddings[0]