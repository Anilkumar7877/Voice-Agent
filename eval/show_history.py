"""
Prints the aggregate eval metrics for every commit you've run the eval
against — this is the actual "measured improvement across N iterations"
view your resume line depends on. Run this after you've run eval/runner.py
at least twice (e.g. before and after a change) to see the comparison.

Run: python eval/show_history.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import results_db


def main():
    summary = results_db.get_summary_by_commit()

    if not summary:
        print("No eval runs recorded yet. Run `python eval/runner.py` first.")
        return

    print(f"{'Commit':<12}{'Cases':<8}{'Tool Acc':<12}{'Task Success':<15}{'Avg Latency':<12}")
    print("-" * 60)
    for row in summary:
        print(
            f"{row['git_commit']:<12}"
            f"{row['num_cases']:<8}"
            f"{row['tool_accuracy'] * 100:.0f}%{'':<8}"
            f"{row['task_success_rate'] * 100:.0f}%{'':<11}"
            f"{row['avg_latency']:.2f}s"
        )

    if len(summary) >= 2:
        first, last = summary[0], summary[-1]
        print("\nChange from first to most recent recorded run:")
        print(f"  Tool accuracy:  {first['tool_accuracy']*100:.0f}% -> {last['tool_accuracy']*100:.0f}%")
        print(f"  Task success:   {first['task_success_rate']*100:.0f}% -> {last['task_success_rate']*100:.0f}%")
        print(f"  Avg latency:    {first['avg_latency']:.2f}s -> {last['avg_latency']:.2f}s")


if __name__ == "__main__":
    main()
