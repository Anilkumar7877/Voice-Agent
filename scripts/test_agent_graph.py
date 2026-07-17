"""
Test the agent's tool-routing logic directly, via text — no mic/STT/TTS
needed. This is the fast iteration loop: confirm the router picks the right
tool and the synthesizer gives a good answer, BEFORE wiring this into the
full voice pipeline (that's the next step after this works).

Run: python scripts/test_agent_graph.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.config import config
from orchestrator.agent.graph import VoiceAgent

TEST_QUESTIONS = [
    "How many vacation days do I get per year?",           # should route to rag
    "What's the weather like in Tokyo right now?",          # should route to web_search
    "Hey, how's it going?",                                  # should route to direct
    "What's the process for expense reimbursement?",         # should route to rag
]


def main():
    config.validate_agent()
    agent = VoiceAgent()

    for question in TEST_QUESTIONS:
        print(f"\n{'=' * 60}")
        print(f"Q: {question}")
        result = agent.run(question)
        print(f"Tool used: {result['tool_choice']}")
        if result["sources"]:
            print(f"Sources: {result['sources']}")
        print(f"A: {result['final_answer']}")


if __name__ == "__main__":
    main()
