"""
Stores eval results tagged by git commit hash + timestamp, so you can
answer "did my change actually help?" with real numbers instead of a
feeling. Each eval run inserts one row per test case, plus you can query
aggregates per commit for the before/after comparison your resume line
depends on.
"""

import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "results.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT NOT NULL,
    git_commit TEXT NOT NULL,
    test_case_id TEXT NOT NULL,
    question TEXT NOT NULL,
    expected_tool TEXT NOT NULL,
    actual_tool TEXT NOT NULL,
    tool_correct INTEGER NOT NULL,
    task_score TEXT NOT NULL,
    task_reasoning TEXT,
    actual_answer TEXT,
    latency_seconds REAL NOT NULL
);
"""


def get_git_commit() -> str:
    """Returns the short commit hash of the current repo state, or 'uncommitted' if not a git repo / no commits yet."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return "uncommitted"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.commit()
    conn.close()


def save_result(test_case_id, question, expected_tool, actual_tool, tool_correct,
                 task_score, task_reasoning, actual_answer, latency_seconds, git_commit):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO eval_runs
           (run_timestamp, git_commit, test_case_id, question, expected_tool,
            actual_tool, tool_correct, task_score, task_reasoning, actual_answer, latency_seconds)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            git_commit,
            test_case_id,
            question,
            expected_tool,
            actual_tool,
            int(tool_correct),
            task_score,
            task_reasoning,
            actual_answer,
            latency_seconds,
        ),
    )
    conn.commit()
    conn.close()


def get_summary_by_commit():
    """Aggregate metrics grouped by commit — this is the table that answers
    'did this change actually help.'"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
            git_commit,
            MIN(run_timestamp) as first_run,
            COUNT(*) as num_cases,
            AVG(tool_correct) as tool_accuracy,
            AVG(CASE WHEN task_score = 'correct' THEN 1.0
                     WHEN task_score = 'partial' THEN 0.5
                     WHEN task_score = 'not_graded' THEN NULL
                     ELSE 0.0 END) as task_success_rate,
            AVG(latency_seconds) as avg_latency
        FROM eval_runs
        GROUP BY git_commit
        ORDER BY first_run ASC
        """
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
