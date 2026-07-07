"""
Text-to-speech client using Cartesia Sonic.

Phase 1: batch only — you give it a complete string, it gives you back a
complete WAV file. Streaming playback (audio starts before the full sentence
is synthesized) is Phase 2 — Cartesia's websocket API is what enables that,
and we'll switch to it then.
"""

from cartesia import Cartesia
from orchestrator.config import config


class TTSClient:
    def __init__(self):
        self.client = Cartesia(api_key=config.CARTESIA_API_KEY)

    def synthesize(self, text: str, output_path: str = "output.wav") -> str:
        """
        Convert text to speech and save it as a WAV file.

        text: what to say.
        output_path: where to save the resulting audio.
        Returns: the output_path, for convenience chaining.
        """
        audio_generator = self.client.tts.bytes(
            model_id=config.CARTESIA_TTS_MODEL,
            transcript=text,
            voice={"mode": "id", "id": config.CARTESIA_VOICE_ID},
            output_format={
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": 44100,
            },
        )

        with open(output_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)

        return output_path
