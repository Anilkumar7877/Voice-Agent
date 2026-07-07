"""
Quickest possible sanity check: does the LLM respond, and does Cartesia
turn that response into audio? No microphone involved — this isolates
two of the three pipeline stages so if something's broken, you know it's
not a mic/audio-capture issue.

Run: python scripts/test_llm_tts.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.config import config
from orchestrator.agent.llm_client import LLMClient
from orchestrator.audio.tts_client import TTSClient


def main():
    config.validate()

    question = "What's a good way to stay focused while studying?"
    print(f"User: {question}")

    llm = LLMClient()
    answer = llm.get_response(question)
    print(f"Agent: {answer}")

    tts = TTSClient()
    output_path = tts.synthesize(answer, output_path="output.wav")
    print(f"\nSaved audio to {output_path} — play it to confirm Cartesia is working.")


if __name__ == "__main__":
    main()
