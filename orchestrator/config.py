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

    # --- Tavily (web search tool) ---
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # --- Chroma (RAG tool) ---
    # This is a DUMMY knowledge base for Phase 3 dev — swap CHROMA_PERSIST_DIR /
    # CHROMA_COLLECTION_NAME to point at your real Document Q&A project's store
    # when you're ready. Nothing else in rag_tool.py needs to change.
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "dummy_kb")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

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

    @classmethod
    def validate_agent(cls):
        """Extra validation for Phase 3+ — call this from agent-related scripts."""
        cls.validate()
        if not cls.TAVILY_API_KEY:
            raise RuntimeError(
                "Missing TAVILY_API_KEY — needed for the web_search tool. "
                "Get one free at https://app.tavily.com and add it to .env"
            )


config = Config()
