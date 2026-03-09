import os

from dotenv import load_dotenv
from openai import OpenAI

from config import GEN_MODEL, MAX_CONTEXT_CHARS
from rag.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


load_dotenv()


def get_client():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set. Put it in .env or export it in the shell.")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def format_context(docs):
    blocks = []

    for i, doc in enumerate(docs, start=1):
        text = doc.page_content[:MAX_CONTEXT_CHARS].strip()

        blocks.append(
            f"[{i}] date={doc.metadata.get('date', '')} | "
            f"platform={doc.metadata.get('platform', '')} | "
            f"rerank_score={doc.metadata.get('rerank_score', '')}\n"
            f"url={doc.metadata.get('post_url', '')}\n"
            f"text={text}"
        )

    return "\n\n".join(blocks)


def generate_answer(question, docs, time_range="", model=GEN_MODEL):
    if not docs:
        return "No retrieved documents. Cannot generate grounded answer."

    client = get_client()
    context = format_context(docs)

    prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        time_range=time_range,
        context=context,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content