"""
Minimal LLM client — plain question-answer, no tool-calling, no LangGraph yet.

Phase 3 will replace this with a LangGraph agent that can route to
RAG/web-search/calendar tools. Keeping it dead simple here so you can verify
the STT -> LLM -> TTS round trip works before adding agent complexity on top.
"""

from groq import Groq
from orchestrator.config import config

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep answers short and conversational "
    "— you are being spoken aloud through text-to-speech, so avoid long lists, "
    "markdown formatting, or anything that doesn't sound natural when read out loud."
)


class LLMClient:
    def __init__(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)

    def get_response(self, user_text: str) -> str:
        """
        Send user text to the LLM and return its reply as a plain string.
        """
        completion = self.client.chat.completions.create(
            model=config.GROQ_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
            max_tokens=300,
        )
        return completion.choices[0].message.content.strip()
