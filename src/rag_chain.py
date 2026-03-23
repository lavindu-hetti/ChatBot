import json
from urllib import request
from urllib.error import URLError

from src.retriever import retrieve_similar_chunks


def build_context(results) -> str:
    context_parts = []
    for i, doc in enumerate(results, start=1):
        page = doc.metadata.get("page", "Unknown")
        content = doc.page_content.strip()
        context_parts.append(f"Source {i} | Page {page}\n{content}")
    return "\n\n".join(context_parts)


def build_sources(results) -> str:
    lines = []
    for i, doc in enumerate(results, start=1):
        page = doc.metadata.get("page", "Unknown")
        preview = doc.page_content[:180].replace("\n", " ").strip()
        lines.append(f"- Source {i} (Page {page}): {preview}...")
    return "\n".join(lines)


def call_ollama(prompt: str, model_name: str, ollama_url: str = "http://127.0.0.1:11434/api/generate") -> str:
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")

    req = request.Request(
        ollama_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(f"Could not reach Ollama: {exc}") from exc

    return body.get("response", "").strip()


def build_rag_prompt(user_question: str, context: str) -> str:
    return f"""You are a helpful PDF question-answering assistant.
Answer the question using only the provided context from the PDF.
If the answer is not in the context, say you could not find it in the PDF.
Be concise and accurate. When useful, mention the page numbers from the sources.

Context:
{context}

Question:
{user_question}

Answer:
"""


def answer_with_retrieval(vectorstore, prompt: str, model_name: str, k: int = 3) -> str:
    results = retrieve_similar_chunks(vectorstore, prompt, k=k)
    context = build_context(results)
    rag_prompt = build_rag_prompt(prompt, context)
    answer = call_ollama(rag_prompt, model_name=model_name)
    sources = build_sources(results)
    return f"{answer}\n\n**Sources**\n{sources}"
