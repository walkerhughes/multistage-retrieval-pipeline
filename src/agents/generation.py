import os
import asyncio
import httpx
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable

from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = str(os.getenv("API_BASE_URL"))


@traceable(name="retrieval")
async def query_database(query: str = ""):
    """Query the retrieval API with tracing.

    Args:
        query: Search query string

    Returns:
        List of retrieved chunks with metadata
    """
    async with httpx.AsyncClient(base_url=API_BASE_URL) as client:
        resp = await client.post("api/retrieval/query", json={
            "query": query,
            "max_returned": 10,
            "mode": "hybrid",
            "operator": "or",
            "fts_candidates": 100
        })
        resp.raise_for_status()
        return resp.json()["chunks"]


client = wrap_openai(OpenAI())  # LangSmith automatically tracks costs via wrap_openai


@traceable(name="vanilla_rag")
async def rag(question: str) -> str:
    """Run RAG pipeline with full tracing.

    LangSmith automatically tracks:
    - Token usage (input/output tokens)
    - Cost (via wrap_openai)
    - Latency for each traced function
    - Model information

    Args:
        question: User question to answer

    Returns:
        Generated answer string
    """
    docs = await query_database(query=question)
    final = ""
    for chunk in docs:
        final += f"""
        Title: {chunk["metadata"]["title"]}
        Text Quotation: {chunk["text"]}
        \n
        """
    system_message = (
        "Answer the user's question using only the provided information below:\n"
        + final
    )
    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
    )
    return str(resp.choices[0].message.content)

if __name__ == "__main__":
    import time
    start = time.time()
    response = asyncio.run(rag("What have Dwarkesh's guests said about the timeline for when we will reach AGI?"))
    total = time.time() - start

    print(response)
    print(f"\nTotal time elapsed in seconds: {total}")
    print("\nCheck LangSmith dashboard for detailed traces including:")
    print("- Token usage (input/output)")
    print("- Cost tracking")
    print("- Latency breakdown")
    print("- Retrieved chunks")