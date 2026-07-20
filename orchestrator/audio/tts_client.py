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

    def stream_and_play(self, text_chunks, interrupt_controller=None) -> bool:
        """
        Take an iterable of text chunks (e.g. from text_chunker.chunk_stream)
        and speak them as they arrive, playing audio in near-real-time
        instead of waiting for the full response to be synthesized.

        interrupt_controller: if provided, checked after every audio chunk
        is written. If set, playback stops immediately — we close the
        output stream and the websocket context rather than continuing to
        play or generate audio nobody will hear. This is the actual
        barge-in mechanism: the moment VAD detects the user talking, the
        orchestrator calls interrupt_controller.trigger(), and this loop
        notices within one audio chunk (tens of milliseconds), not at the
        end of the current sentence.

        Returns: True if playback completed normally, False if it was
        interrupted partway through.

        Uses Cartesia's *context* API (client.tts.websocket() -> ws.context())
        which is the real multi-turn streaming interface — NOT ws.send()
        directly. ws.send() is a backwards-compatible convenience wrapper
        that always sends continue_=False internally, so passing continue_
        yourself collides with it (this is exactly the TypeError you'll hit
        if you try). Using ctx.send()/ctx.receive() on a context object gives
        us real control over continue_ so we can keep one logical utterance
        open across multiple text chunks.

        Verified against cartesia==3.3.0 — if your installed version differs
        (`pip show cartesia`), the shape of WebsocketResponse events (chunk
        vs done vs error) is the most likely thing to have changed.
        """
        import sounddevice as sd
        import numpy as np
        import uuid

        sample_rate = 44100
        stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16")
        stream.start()

        ws = self.client.tts.websocket()
        context_id = str(uuid.uuid4())
        ctx = ws.context(context_id)

        completed_normally = True

        try:
            chunks = list(text_chunks)
            for i, text_chunk in enumerate(chunks):
                if interrupt_controller and interrupt_controller.is_set():
                    completed_normally = False
                    break

                is_last = i == len(chunks) - 1
                ctx.send(
                    model_id=config.CARTESIA_TTS_MODEL,
                    transcript=text_chunk,
                    voice={"mode": "id", "id": config.CARTESIA_VOICE_ID},
                    output_format={
                        "container": "raw",
                        "encoding": "pcm_s16le",
                        "sample_rate": sample_rate,
                    },
                    continue_=not is_last,
                )

            if completed_normally:
                for event in ctx.receive():
                    if interrupt_controller and interrupt_controller.is_set():
                        completed_normally = False
                        break
                    if event.type == "error":
                        raise RuntimeError(f"Cartesia TTS error: {getattr(event, 'error', event)}")
                    if event.type == "chunk":
                        audio_array = np.frombuffer(event.audio, dtype=np.int16)
                        stream.write(audio_array)
                        if interrupt_controller and interrupt_controller.is_set():
                            completed_normally = False
                            break
                    if event.type == "done":
                        break
        finally:
            stream.stop()
            stream.close()
            ws.close()

        return completed_normally

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
