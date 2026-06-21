from __future__ import annotations

from typing import List

from openai import OpenAI

from app.config import get_settings


class OpenAIService:
    def __init__(self) -> None:
        settings = get_settings()
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        # You can change the default model here if desired
        self.model = "gpt-4.1-mini"

    def ask_with_context(self, context: str, question: str) -> str:
        """
        Call OpenAI Chat Completion with the given context and question.
        """
        messages: List[dict] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the user's question using "
                    "ONLY the provided context. If the answer is not contained in "
                    "the context, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"<context>\n{context}\n</context>\n\nQuestion:\n{question}",
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content or ""

