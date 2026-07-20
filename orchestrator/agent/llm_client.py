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
        Kept for Phase 1 compatibility / quick tests. Prefer stream_response
        for anything real-time, since this waits for the full generation.
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

    def stream_response(self, user_text: str, interrupt_controller=None):
        """
        Yield the LLM's reply token-by-token as it's generated, instead of
        waiting for the full response. This is what makes low end-to-end
        latency possible — we can start sending text to TTS before the LLM
        has finished "thinking."

        interrupt_controller: if provided, checked after every token —
        stops yielding immediately (and the underlying stream is abandoned)
        if the user has started talking over the agent. This is the
        cooperative-cancellation pattern: we can't force-cancel the
        in-flight Groq HTTP stream from outside, so we simply stop
        consuming it and let it close naturally.

        Yields: str chunks (usually a few characters / one token at a time).
        """
        stream = self.client.chat.completions.create(
            model=config.GROQ_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
            max_tokens=300,
            stream=True,
        )
        for chunk in stream:
            if interrupt_controller and interrupt_controller.is_set():
                return
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
