"""Shared helpers: trace logging, text detection, quiz grading."""
import os
import re
import logging
import json
from typing import Any, List, Dict

# ---------------------------------------------------------------------------
# Trace logging
# ---------------------------------------------------------------------------

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

trace_logger = logging.getLogger("SocrAItes.Trace")
trace_logger.setLevel(logging.INFO)
if not trace_logger.handlers:
    fh = logging.FileHandler(os.path.join(LOG_DIR, "agent_trace.log"), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    trace_logger.addHandler(fh)


def _log_trace(
    step: str,
    purpose: str = "",
    inputs: dict = None,
    prompt_details: Any = None,
    response: Any = None,
    decision: Any = None,
):
    border = "=" * 80
    trace_logger.info(border)
    trace_logger.info(f"NODE: {step.upper()}")
    trace_logger.info(border)

    if purpose:
        trace_logger.info(f"[PURPOSE]\n  {purpose}\n")

    if inputs:
        trace_logger.info("[INPUT CONTEXT]")
        for k, v in inputs.items():
            trace_logger.info(f"  * {k}: {v}")
        trace_logger.info("")

    if prompt_details:
        trace_logger.info("[LLM PROMPT / REQUEST CONTENT]")
        if isinstance(prompt_details, str):
            indented = "\n".join(f"    {line}" for line in prompt_details.split("\n"))
            trace_logger.info(f"{indented}\n")
        elif isinstance(prompt_details, dict):
            for pk, pv in prompt_details.items():
                trace_logger.info(f"  - {pk}:")
                indented = "\n".join(f"      {line}" for line in str(pv).split("\n"))
                trace_logger.info(indented)
            trace_logger.info("")
        else:
            trace_logger.info(f"  {prompt_details}\n")

    if response is not None:
        trace_logger.info("[LLM RESPONSE]")
        trace_logger.info("-" * 80)
        indented = "\n".join(f"  {line}" for line in str(response).split("\n"))
        trace_logger.info(indented)
        trace_logger.info("-" * 80)
        trace_logger.info("")

    if decision:
        trace_logger.info("[DECISION & STATE MUTATION]")
        if isinstance(decision, dict):
            for dk, dv in decision.items():
                trace_logger.info(f"  * {dk}: {dv}")
        else:
            trace_logger.info(f"  * Result: {decision}")

    trace_logger.info(border + "\n\n")


# ---------------------------------------------------------------------------
# Text detection helpers
# ---------------------------------------------------------------------------

_ANSWER_PATTERN = re.compile(r"(\d+)\s*[:：]\s*([A-Da-d])", re.UNICODE)


def _is_quiz_answer(text: str) -> bool:
    return len(_ANSWER_PATTERN.findall(text)) >= 2


def _detect_frustration(text: str) -> bool:
    keywords = [
        "모르겠", "모르겠음", "모르겠어", "모르겠다", "모르겠는데",
        "어렵", "어려워", "어려움", "어렵다", "어려운데",
        "힘들", "힘들어", "힘들다", "힘듦", "힘든데",
        "포기", "못하겠", "못하겠어", "못하겠다", "못하겠음",
        "답답", "헷갈려", "헷갈림", "이해 안", "이해가 안", "이해 안 됨",
        "그냥 알려줘", "답 알려줘", "답을 알려", "어려운",
    ]
    clean = text.replace(" ", "").lower()
    return any(k.replace(" ", "").lower() in clean for k in keywords)


def _is_summary_request(text: str) -> bool:
    keywords = ["요약", "정리", "summary", "summarize", "summarise", "outline"]
    clean = text.replace(" ", "").lower()
    return any(k in clean for k in keywords)


def _is_last_message_quiz_prompt(history: List[Dict[str, Any]]) -> bool:
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return "퀴즈" in content and "답안은 예:" in content
    return False


def _format_history(messages: List[Dict[str, Any]]) -> str:
    lines = []
    for msg in messages[:-1]:
        role = "Student" if msg["role"] == "user" else "Tutor"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quiz helpers
# ---------------------------------------------------------------------------

def _grade_quiz(user_text: str, quiz_items: List[Dict[str, Any]]) -> str:
    submissions = {int(q): v.upper() for q, v in _ANSWER_PATTERN.findall(user_text)}
    total = len(quiz_items)
    correct = 0
    lines = ["## 채점 결과", ""]

    for idx, item in enumerate(quiz_items, start=1):
        correct_ans = item.get("answer", "").upper()
        user_ans = submissions.get(idx, "?")
        options = item.get("options", [])
        correct_text = options[ord(correct_ans) - ord("A")] if correct_ans and options else correct_ans

        if user_ans == correct_ans:
            correct += 1
            mark = "✅"
        else:
            mark = "❌"

        lines.append(f"{mark} {idx}번: 제출={user_ans} / 정답={correct_ans}")
        if user_ans != correct_ans:
            lines.append(f"   → 정답: {correct_ans}. {correct_text}")

    score_pct = int(correct / total * 100) if total else 0
    lines += ["", f"**{total}문항 중 {correct}문항 정답 ({score_pct}점)**"]

    if score_pct == 100:
        lines.append("완벽해요! 다음 개념으로 넘어가볼까요?")
    elif score_pct >= 60:
        lines.append("잘 했어요! 틀린 문항을 다시 한번 살펴보세요.")
    else:
        lines.append("조금 더 복습이 필요해요. 틀린 개념을 약점으로 저장해드릴까요?")

    return "\n".join(lines)


def _format_quiz_response(
    tool_results: List[Dict[str, Any]],
) -> tuple[str | None, List[Dict[str, Any]]]:
    for result in tool_results:
        if result.get("tool") != "generate_quiz" or not result.get("ok"):
            continue
        output = result.get("output") or {}
        quiz_items = output.get("quiz") if isinstance(output, dict) else None
        if not isinstance(quiz_items, list) or not quiz_items:
            continue

        source = output.get("source", "template")
        header = "퀴즈 %d문항을 준비했어요 (출처: %s). 한 번에 풀어보세요." % (
            len(quiz_items),
            "강의 자료 기반" if source == "llm_rag" else "기본 템플릿",
        )
        lines = [header, ""]
        for idx, item in enumerate(quiz_items, start=1):
            question = item.get("question", "질문")
            options = item.get("options", [])
            lines.append(f"{idx}. {question}")
            if isinstance(options, list) and len(options) >= 4:
                lines += [f"A. {options[0]}", f"B. {options[1]}", f"C. {options[2]}", f"D. {options[3]}"]
            lines.append("")
        lines.append("답안은 예: 1:A, 2:B, 3:A, 4:C, 5:D 형태로 보내주세요.")
        return "\n".join(lines), quiz_items
    return None, []
