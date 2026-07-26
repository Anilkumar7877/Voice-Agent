"""
LLM-as-judge scorer: is the agent's answer actually correct, given the
ground truth? This is necessarily fuzzier than tool_accuracy_scorer — two
answers can be worded completely differently and both be correct — so we
use an LLM to judge semantic correctness rather than string matching.

Uses a separate, low-temperature call with a strict output format (just a
score + one-line reasoning) so results are as consistent as possible across
repeated runs — LLM judges are not perfectly deterministic, which is worth
knowing rather than treating this score as ground truth itself.
"""

import json
from groq import Groq
from orchestrator.config import config

JUDGE_PROMPT = """You are evaluating whether an AI assistant's answer is \
correct, given a reference/ground-truth answer. Score it as one of:
- "correct": the answer conveys the same key facts as the ground truth, \
even if worded differently
- "partial": the answer is roughly right but missing or slightly wrong on \
a key detail
- "incorrect": the answer is wrong, contradicts the ground truth, or \
doesn't address the question

Question: {question}

Ground truth: {ground_truth}

Assistant's answer: {actual_answer}

Respond with ONLY a JSON object, nothing else, in this exact format:
{{"score": "correct" | "partial" | "incorrect", "reasoning": "one sentence explaining why"}}"""


def score_task_success(question: str, ground_truth: str, actual_answer: str) -> dict:
    client = Groq(api_key=config.GROQ_API_KEY)

    completion = client.chat.completions.create(
        model=config.GROQ_LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    question=question, ground_truth=ground_truth, actual_answer=actual_answer
                ),
            }
        ],
        temperature=0,
        max_tokens=150,
    )

    raw = completion.choices[0].message.content.strip()

    try:
        # Strip markdown code fences if the model adds them despite instructions
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        return {"score": parsed.get("score", "incorrect"), "reasoning": parsed.get("reasoning", "")}
    except (json.JSONDecodeError, AttributeError):
        # Defensive fallback — don't crash the whole eval run over one
        # malformed judge response, just flag it clearly in the results.
        return {"score": "unparseable", "reasoning": f"Judge response could not be parsed: {raw[:100]}"}
