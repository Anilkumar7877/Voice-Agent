"""
Run this once (and again any time you change DUMMY_DOCS in rag_tool.py) to
build/rebuild the dummy knowledge base.

Run: python scripts/seed_dummy_kb.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.agent.tools.rag_tool import RAGTool


def main():
    print("Building dummy knowledge base...")
    rag_tool = RAGTool()
    rag_tool.seed()
    print("Done. Try a query with scripts/test_agent_graph.py")


if __name__ == "__main__":
    main()
