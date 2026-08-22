"""
Ingests the ai4bharat/MSMARCO-XI dataset (the dataset required by the task)
into the existing Document/Chunk pgvector tables.

Usage:
    python -m backend.scripts.ingest_msmarco --language hi --limit 500

Requires: pip install datasets   (not in requirements.txt by default since
it's a large, one-off dependency only needed for this ingestion step)

REAL DATASET SCHEMA (confirmed from the HF dataset card):
    {
        "query": "<translated query in target language>",
        "Answer": "<translated answer>",
        "query_id": 1185869,
        "query_type": "DESCRIPTION",
        "passages": {
            "is_selected": [1, 0, 0, ...],
            "English_passages": ["...", "...", ...],
            "Translated_passages": ["...", "...", ...],
        },
        "Eng_Query": "<original English query>",
        "Eng_Answer": "<original English answer>",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
    }

We index `English_passages` (not `Translated_passages`) because the existing
embedder uses Cohere's embed-english-v3.0 — an English-only embedding model.
If you want true multilingual retrieval on the translated text, swap in a
multilingual embedding model first; indexing Indic-script text through an
English embedder will produce poor/meaningless vectors.

Chunking: every passage is run through TWO strategies (sentence-aware, and
semantic for longer passages), tagged with `chunk_strategy` so you can filter
or compare them later — satisfying the "vast chunking" requirement.

DOWNLOAD NOTE: this downloads the full per-language parquet file (several GB)
once — Hugging Face doesn't support fetching only specific rows without a
full download. The download is resumable (safe to rerun if your connection
drops) and only ever runs once since it's cached locally afterward. We then
read only the rows we need in small batches via pyarrow, which is what avoids
the earlier out-of-memory crash from loading the whole file at once.

Requires: pip install pyarrow huggingface_hub
"""
import argparse
import itertools
import sys

from backend.db import SessionLocal
from backend.models.document import Document, Chunk
from backend.services.chunker import sentence_chunk_text, semantic_chunk_text
from backend.services.embedder import embed_chunks

# The dataset does NOT have per-language "configs" you select by name (that was
# wrong in an earlier version of this script — HF only reports one config,
# "default"). Instead, each language lives in its own file, named per the
# dataset card's file table. We load that specific file directly.
LANGUAGE_FILE_PREFIX = {
    "as": "asm", "bn": "ben", "gu": "gu", "hi": "hin", "kn": "kan",
    "ml": "mal", "mr": "mar", "ne": "nep", "or": "or", "pa": "pan",
    "sa": "san", "ta": "tam", "te": "tel", "ur": "urd",
}


def _iter_rows(language: str, limit: int | None):
    """
    Downloads the parquet file once (hf_hub_download resumes automatically if
    a download drops partway — it won't restart from zero on a flaky
    connection) into the local HF cache, then reads only the row groups it
    needs via pyarrow — never loading the whole multi-GB file into memory.
    This replaces an earlier streaming=True approach that made many small
    HTTP range requests and kept hitting "connection reset by peer" on
    unstable connections.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    prefix = LANGUAGE_FILE_PREFIX[language]
    data_file = f"train/{prefix}train.parquet"

    print(f"Downloading {data_file} (resumable — safe to retry if your connection drops "
          f"partway; it picks up from the cache, it won't restart from 0) ...")
    local_path = hf_hub_download(
        repo_id="ai4bharat/MSMARCO-XI", repo_type="dataset", filename=data_file,
    )
    print(f"Download complete: {local_path}")

    pf = pq.ParquetFile(local_path)
    count = 0
    for batch in pf.iter_batches(batch_size=200):  # reads row-group-sized batches, not the whole file
        for row in batch.to_pylist():
            yield row
            count += 1
            if limit and count >= limit:
                return


def ingest(language: str = "hi", limit: int | None = 500, batch_flush: int = 200):
    try:
        import pyarrow  # noqa: F401
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("Missing dependency: run `pip install pyarrow huggingface_hub` first "
              "(huggingface_hub normally comes with `datasets`, but install it explicitly if needed).")
        sys.exit(1)

    if language not in LANGUAGE_FILE_PREFIX:
        print(f"Unknown language '{language}'. Valid codes: {', '.join(LANGUAGE_FILE_PREFIX)}")
        sys.exit(1)

    try:
        dataset = _iter_rows(language, limit)
    except Exception as e:
        print(f"Could not open the dataset file: {e}")
        print("If this is a network/DNS error (e.g. 'nodename nor servname provided' or "
              "'Connection reset by peer'), that's your machine's connection to huggingface.co, "
              "not a code bug — check Wi-Fi stability or any VPN/firewall, then just rerun the "
              "same command; the download resumes instead of starting over.")
        sys.exit(1)

    db = SessionLocal()

    doc = Document(filename=f"msmarco-xi-{language}", total_chunks=0)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    pending_chunks = []
    total_inserted = 0
    skipped = 0
    examples_seen = 0

    def flush():
        nonlocal pending_chunks, total_inserted
        if not pending_chunks:
            return
        texts = [c["content"] for c in pending_chunks]
        embeddings = embed_chunks(texts)
        for c, emb in zip(pending_chunks, embeddings):
            db.add(Chunk(
                document_id=doc.id,
                content=c["content"],
                embedding=emb,
                chunk_strategy=c["chunk_strategy"],
                language=c["language"],
                source_query_id=c["source_query_id"],
            ))
        db.commit()
        total_inserted += len(pending_chunks)
        print(f"  ...inserted {total_inserted} chunks so far ({examples_seen} examples processed)")
        pending_chunks = []

    for idx, example in enumerate(dataset):
        examples_seen += 1
        query_id = str(example.get("query_id", idx))
        passages_field = example.get("passages") or {}
        english_passages = passages_field.get("English_passages") or []

        if not english_passages:
            skipped += 1
            continue

        for passage_text in english_passages:
            if not passage_text or not passage_text.strip():
                continue

            # Strategy 1: sentence-aware grouping — cheap, always applied
            for c in sentence_chunk_text(passage_text, max_words=120, overlap_sentences=1):
                pending_chunks.append({
                    "content": c, "chunk_strategy": "sentence",
                    "language": "en", "source_query_id": query_id,
                })

            # Strategy 2: semantic splitting — only worth it on longer passages
            if len(passage_text.split()) > 150:
                for c in semantic_chunk_text(passage_text, embed_fn=embed_chunks):
                    pending_chunks.append({
                        "content": c, "chunk_strategy": "semantic",
                        "language": "en", "source_query_id": query_id,
                    })

        if len(pending_chunks) >= batch_flush:
            flush()

    flush()

    doc.total_chunks = total_inserted
    db.commit()
    doc_id = doc.id  # grab this before closing the session — accessing it after close() fails
    db.close()
    print(f"Done. Indexed {total_inserted} chunks from {examples_seen} examples "
          f"({skipped} examples skipped — no passages) into document id={doc_id}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="hi",
                         help="Dataset config: as, bn, gu, hi, kn, ml, mr, ne, or, pa, sa, ta, te, ur "
                              "(this only affects which query-set is pulled in; passages indexed are English)")
    parser.add_argument("--limit", type=int, default=500, help="Max number of examples to ingest (0 = all)")
    args = parser.parse_args()
    ingest(language=args.language, limit=(args.limit or None))
