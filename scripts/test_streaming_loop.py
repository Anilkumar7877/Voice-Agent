"""
Phase 2 full loop: silence-triggered recording -> batch STT (Groq Whisper) ->
STREAMING LLM tokens -> STREAMING TTS playback.

The STT step is still batch (see README note on why Groq doesn't support
streaming STT). The win here is in the LLM + TTS stages: the agent starts
speaking before it's finished generating the full answer.

This script also prints basic timing so you can start noticing the
latency numbers your eval harness will formalize later:
- time to transcript (STT)
- time to first token (LLM)
- time to first audio (TTS) <- the metric that matters most for "feels real-time"

Run: python scripts/test_streaming_loop.py
"""

import sys
import os
import time
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.io.wavfile import write as wav_write

from orchestrator.config import config
from orchestrator.audio.stt_client import STTClient
from orchestrator.audio.tts_client import TTSClient
from orchestrator.audio.text_chunker import chunk_stream
from orchestrator.audio.silence_detector import record_until_silence
from orchestrator.agent.llm_client import LLMClient


def main():
    config.validate()

    stt = STTClient()
    llm = LLMClient()
    tts = TTSClient()

    # 1. Record until the user stops talking
    audio_data = record_until_silence(sample_rate=config.SAMPLE_RATE)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_write(tmp.name, config.SAMPLE_RATE, audio_data)
        input_path = tmp.name

    # 2. Transcribe (batch — timed so you can see this stage's cost)
    t0 = time.time()
    user_text = stt.transcribe(input_path)
    t_transcript = time.time() - t0
    print(f"You said: {user_text}  [STT took {t_transcript:.2f}s]")

    if not user_text:
        print("No speech detected — try again and speak clearly into the mic.")
        return

    # 3. Stream LLM tokens, chunk into sentences, stream to TTS as they arrive
    print("Agent (streaming): ", end="", flush=True)

    t_llm_start = time.time()
    first_token_time = None

    def timed_token_stream():
        nonlocal first_token_time
        for token in llm.stream_response(user_text):
            if first_token_time is None:
                first_token_time = time.time()
            print(token, end="", flush=True)
            yield token

    text_chunks = chunk_stream(timed_token_stream())

    # tts.stream_and_play consumes the generator, which pulls tokens from the
    # LLM as needed — so LLM generation and TTS playback are pipelined, not
    # sequential (i.e. we don't wait for the full answer before speaking).
    tts.stream_and_play(text_chunks)

    print()
    if first_token_time:
        print(f"[Time to first LLM token: {first_token_time - t_llm_start:.2f}s]")


if __name__ == "__main__":
    main()
