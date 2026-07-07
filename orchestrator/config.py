"""
Central config for the voice agent.

Why this file exists: every provider/model choice lives here, not scattered
across client files. When you swap STT/TTS providers later (e.g. Deepgram
instead of Groq Whisper) for your eval harness comparisons, you change one
file, not five.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- Groq (LLM + STT) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
    GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

    # --- Cartesia (TTS) ---
    CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
    CARTESIA_TTS_MODEL = os.getenv("CARTESIA_TTS_MODEL", "sonic-2")
    # Default voice ID is Cartesia's public demo voice. Browse more at
    # play.cartesia.ai -> Voice Library, and copy the ID of one you like into .env
    CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091")

    # --- Audio recording settings (used by test_full_loop.py) ---
    SAMPLE_RATE = 16000  # Whisper expects 16kHz mono
    RECORD_SECONDS = 5

    @classmethod
    def validate(cls):
        missing = []
        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not cls.CARTESIA_API_KEY:
            missing.append("CARTESIA_API_KEY")
        if missing:
            raise RuntimeError(
                f"Missing required env vars: {', '.join(missing)}. "
                f"Did you copy .env.example to .env and fill it in?"
            )


config = Config()
