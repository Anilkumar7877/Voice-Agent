"""
Cooperative cancellation signal — deliberately NOT built on
asyncio.Task.cancel(). Cartesia's and Groq's Python SDKs are synchronous;
there's no `await` point inside their blocking calls for asyncio to
interrupt. The correct pattern for cancelling synchronous work is
cooperative: the long-running loop (TTS playback, LLM token streaming)
checks `.is_set()` after every unit of work (audio chunk, token) and exits
early if told to. This IS how real interruption is done across a sync SDK
boundary — it's not a compromise, it's the standard approach.
"""

import threading


class InterruptController:
    def __init__(self):
        self._event = threading.Event()

    def trigger(self):
        self._event.set()

    def clear(self):
        self._event.clear()

    def is_set(self) -> bool:
        return self._event.is_set()
