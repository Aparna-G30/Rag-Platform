import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def generate_answer(query: str, chunks: list[dict]) -> dict:
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"[Source {i} — Page {chunk['page_number']}]\n"
        context += chunk["content"] + "\n\n"

    prompt = f"""You are a precise document assistant.
Answer the question using ONLY the sources below.
For every claim, cite the source number in brackets like [1].
If the answer is not in the sources, say "I cannot find this in the document."

SOURCES:
{context}

QUESTION: {query}

ANSWER:"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    return {"answer": response.text, "sources_used": len(chunks)}