"""
Speech-to-text client using Groq's hosted Whisper.

Phase 1: batch only — you give it a complete audio file, it gives you back
a complete transcript. No partial/streaming results yet (that's Phase 2,
and it needs a different API shape — Groq's Whisper endpoint here is
request/response, not a websocket).
"""

from groq import Groq
from orchestrator.config import config


class STTClient:
    def __init__(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)

    def transcribe(self, audio_file_path: str) -> str:
        """
        Transcribe a complete audio file (wav/mp3/m4a etc.) to text.

        audio_file_path: path to an audio file on disk.
        Returns: the transcribed text, stripped of leading/trailing whitespace.
        """
        with open(audio_file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                file=audio_file,
                model=config.GROQ_WHISPER_MODEL,
                response_format="text",
            )
        # response_format="text" returns a raw string in some SDK versions
        # and an object with .text in others — handle both defensively.
        if isinstance(transcription, str):
            return transcription.strip()
        return transcription.text.strip()
