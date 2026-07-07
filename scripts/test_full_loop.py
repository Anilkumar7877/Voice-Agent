"""
Full Phase 1 loop: record your voice from the mic, transcribe it, get an
LLM answer, synthesize speech, and play it back.

This is NOT real-time and NOT streaming — it records a fixed window, then
processes it in sequence. That's intentional: get each stage correct first,
add real-time behavior in Phase 2.

Run: python scripts/test_full_loop.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as wav_write

from orchestrator.config import config
from orchestrator.audio.stt_client import STTClient
from orchestrator.audio.tts_client import TTSClient
from orchestrator.agent.llm_client import LLMClient


def record_audio(seconds: int, sample_rate: int, output_path: str = "input.wav") -> str:
    print(f"Recording for {seconds} seconds — speak now...")
    recording = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    wav_write(output_path, sample_rate, recording)
    print("Recording finished.")
    return output_path


def play_audio(path: str):
    from scipy.io.wavfile import read as wav_read

    sample_rate, data = wav_read(path)
    sd.play(data, sample_rate)
    sd.wait()


def main():
    config.validate()

    stt = STTClient()
    llm = LLMClient()
    tts = TTSClient()

    # 1. Record
    input_path = record_audio(config.RECORD_SECONDS, config.SAMPLE_RATE)

    # 2. Transcribe
    print("Transcribing...")
    user_text = stt.transcribe(input_path)
    print(f"You said: {user_text}")

    if not user_text:
        print("No speech detected — try again and speak clearly into the mic.")
        return

    # 3. Get LLM response
    print("Thinking...")
    answer = llm.get_response(user_text)
    print(f"Agent: {answer}")

    # 4. Synthesize speech
    print("Synthesizing speech...")
    output_path = tts.synthesize(answer, output_path="output.wav")

    # 5. Play it back
    print("Playing response...")
    play_audio(output_path)


if __name__ == "__main__":
    main()
