"""
Runs every test case in scenarios.json through the agent, scores each
result, and saves everything to SQLite tagged by the current git commit.

This is the core loop of the eval harness — run this after any change to
the agent (prompt tweaks, model swaps, routing logic changes) to get an
objective before/after comparison instead of a vibe-based "feels better."

Run: python eval/runner.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.config import config
from orchestrator.agent.graph import VoiceAgent
from eval.scorers.tool_accuracy_scorer import score_tool_accuracy
from eval.scorers.llm_judge_scorer import score_task_success
from eval import results_db

SCENARIOS_PATH = os.path.join(os.path.dirname(__file__), "test_cases", "scenarios.json")


def load_scenarios():
    with open(SCENARIOS_PATH) as f:
        return json.load(f)


def main():
    config.validate_agent()
    results_db.init_db()

    git_commit = results_db.get_git_commit()
    print(f"Running eval against commit: {git_commit}\n")

    scenarios = load_scenarios()
    agent = VoiceAgent()

    tool_correct_count = 0
    task_score_counts = {"correct": 0, "partial": 0, "incorrect": 0, "unparseable": 0}
    latencies = []

    for scenario in scenarios:
        print(f"[{scenario['id']}] {scenario['question']}")

        t0 = time.time()
        result = agent.run(scenario["question"])
        latency = time.time() - t0
        latencies.append(latency)

        tool_result = score_tool_accuracy(scenario["expected_tool"], result["tool_choice"])

        if scenario.get("skip_task_scoring"):
            # Time-sensitive content (weather, stock prices, "latest version
            # of X") can't be reliably graded against a fixed ground truth,
            # and an LLM judge shares the same training-cutoff blind spot as
            # the agent itself for anything recent — so we only check that
            # routing was correct, not the content of the answer.
            task_result = {"score": "not_graded", "reasoning": "skip_task_scoring set — content is time-sensitive"}
        else:
            task_result = score_task_success(
                scenario["question"], scenario["ground_truth"], result["final_answer"]
            )

        if tool_result["correct"]:
            tool_correct_count += 1
        task_score_counts[task_result["score"]] = task_score_counts.get(task_result["score"], 0) + 1

        status = "✓" if tool_result["correct"] else "✗"
        print(f"  Tool: {status} (expected={scenario['expected_tool']}, actual={result['tool_choice']})")
        print(f"  Task: {task_result['score']} — {task_result['reasoning']}")
        print(f"  Latency: {latency:.2f}s\n")

        results_db.save_result(
            test_case_id=scenario["id"],
            question=scenario["question"],
            expected_tool=scenario["expected_tool"],
            actual_tool=result["tool_choice"],
            tool_correct=tool_result["correct"],
            task_score=task_result["score"],
            task_reasoning=task_result["reasoning"],
            actual_answer=result["final_answer"],
            latency_seconds=latency,
            git_commit=git_commit,
        )

    n = len(scenarios)
    avg_latency = sum(latencies) / n
    print("=" * 60)
    print(f"SUMMARY — commit {git_commit}")
    print(f"  Tool accuracy: {tool_correct_count}/{n} ({100 * tool_correct_count / n:.0f}%)")
    print(f"  Task success:  {task_score_counts}")
    print(f"  Avg latency:   {avg_latency:.2f}s")
    print("=" * 60)
    print("\nRun `python eval/show_history.py` to compare across commits.")


if __name__ == "__main__":
    main()
