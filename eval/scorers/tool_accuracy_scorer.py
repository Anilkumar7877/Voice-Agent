"""
Simplest scorer in the harness — pure rule-based comparison, no LLM needed.
Deliberately kept dumb: this is the kind of thing that should never need
"judgment," and if it's ever wrong, the bug is in the test case's
expected_tool label, not in needing a smarter scorer.
"""


def score_tool_accuracy(expected_tool: str, actual_tool: str) -> dict:
    correct = expected_tool == actual_tool
    return {
        "correct": correct,
        "expected": expected_tool,
        "actual": actual_tool,
    }
