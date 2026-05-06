"""LLM client config: GPT-5-mini via GitHub Models.

GitHub Models is OpenAI-compatible and free in Actions runners using the
built-in GITHUB_TOKEN (the workflow needs `permissions: models: read`).
"""
import os
from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.3, model: str = "openai/gpt-5-mini") -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url="https://models.github.ai/inference",
        api_key=os.environ["GITHUB_TOKEN"],
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )
