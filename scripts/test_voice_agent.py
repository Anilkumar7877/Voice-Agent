"""
Full voice loop using the AGENT (tool-routing) instead of a plain LLM call.

Flow: silence-triggered recording -> STT -> VoiceAgent.run() (routes to
rag / web_search / direct internally) -> chunk the finished answer into
sentences -> stream those to TTS for playback.

NOTE ON STREAMING: graph.invoke() is blocking — it runs the full
route -> tool -> synthesize sequence and returns the complete answer at
once. So unlike test_streaming_loop.py, the LLM isn't streaming
token-by-token here. What we DO still get: the finished answer is split
into sentences and sent to TTS as separate chunks (via chunk_text), so
playback starts after the first sentence is ready rather than waiting for
one single giant synthesis call. It's a middle ground — not as fast as full
pipelined streaming, but not as slow as Phase 1 either. Making the agent's
own LLM calls stream token-by-token is a natural next refinement once
barge-in (Phase 4) is in place.

Run: python scripts/test_voice_agent.py
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
from orchestrator.audio.text_chunker import chunk_text
from orchestrator.audio.silence_detector import record_until_silence
from orchestrator.agent.graph import VoiceAgent


def main():
    config.validate_agent()

    stt = STTClient()
    tts = TTSClient()
    agent = VoiceAgent()

    audio_data = record_until_silence(sample_rate=config.SAMPLE_RATE)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_write(tmp.name, config.SAMPLE_RATE, audio_data)
        input_path = tmp.name

    t0 = time.time()
    user_text = stt.transcribe(input_path)
    t_transcript = time.time() - t0
    print(f"You said: {user_text}  [STT took {t_transcript:.2f}s]")

    if not user_text:
        print("No speech detected — try again and speak clearly into the mic.")
        return

    print("Agent thinking (routing + retrieving + synthesizing)...")
    t_agent_start = time.time()
    result = agent.run(user_text)
    t_agent = time.time() - t_agent_start

    print(f"[Tool used: {result['tool_choice']}  |  agent took {t_agent:.2f}s]")
    if result["sources"]:
        print(f"Sources: {result['sources']}")
    print(f"Agent: {result['final_answer']}")

    sentence_chunks = chunk_text(result["final_answer"])
    tts.stream_and_play(iter(sentence_chunks))


if __name__ == "__main__":
    main()
