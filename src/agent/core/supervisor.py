"""Core Agent — Supervisor: synthesizes sub-agent outputs into the final answer."""
import json
import logging
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.llm import llm, _get_content
from src.agent.helpers import (
    _log_trace,
    _is_quiz_answer,
    _is_last_message_quiz_prompt,
    _detect_frustration,
    _grade_quiz,
    _format_quiz_response,
)

logger = logging.getLogger("SocrAItes.Agent")


def supervisor(state: AgentState) -> AgentState:
    """Synthesizes sub-agent outputs into the final draft answer.

    Priority:
    1. Quiz grading (pending_quiz + answer submission)
    2. Diagnosis tool results (quiz gen, escape, weakness, schedule)
    3. Socratic Dialogue Agent response
    4. Fallback LLM generation
    """
    logger.info("--- [Supervisor] Synthesis Step ---")
    history = state.get("messages", [])
    last_msg = history[-1]["content"] if history else ""
    pending_quiz = state.get("pending_quiz", [])

    # 1. Quiz grading
    if pending_quiz and _is_quiz_answer(last_msg):
        state["draft_answer"] = _grade_quiz(last_msg, pending_quiz)
        state["pending_quiz"] = []
        state["tool_results"] = []
        logger.info("Quiz graded.")
        _log_trace(step="Supervisor", purpose="Grade quiz submission.", decision="Quiz graded.")
        return state

    # 2. Diagnosis tool results
    tool_results = state.get("diagnosis_result", {}).get("tool_results", [])
    if tool_results:
        quiz_message, quiz_items = _format_quiz_response(tool_results)
        if quiz_message:
            state["draft_answer"] = quiz_message
            state["tool_results"] = tool_results
            state["pending_quiz"] = quiz_items
            logger.info("Quiz formatted from Diagnosis Agent.")
            _log_trace(step="Supervisor", purpose="Format quiz.", decision="pending_quiz set.")
            return state

        is_quiz_pending = len(pending_quiz) > 0 or _is_last_message_quiz_prompt(history[:-1])
        frustration_hint = (
            "IMPORTANT: Student is frustrated during a quiz. Offer a hint and encouragement. "
            "Do NOT call generate_quiz. Tell them they can type '그냥 답 알려줘' to get the answer."
            if is_quiz_pending and _detect_frustration(last_msg)
            else ""
        )
        synthesis_prompt = (
            "You are SocrAItes. Synthesize the tool results into a concise Korean response.\n\n"
            f"Tool Results:\n{json.dumps(tool_results, ensure_ascii=False)}\n\n"
            "Rules:\n"
            "1. If escape_to_answer succeeded, provide a direct, helpful answer in Korean.\n"
            "2. If weakness/schedule was saved, briefly acknowledge and ask one Socratic follow-up.\n"
            "3. Keep it concise and warm.\n"
            f"{frustration_hint}"
        )
        response = llm.invoke([SystemMessage(content=synthesis_prompt), HumanMessage(content="최종 응답을 작성해줘.")])
        state["draft_answer"] = _get_content(response)
        state["tool_results"] = tool_results
        _log_trace(step="Supervisor", purpose="Synthesize tool results.", response=state["draft_answer"])
        return state

    # 3. Socratic Dialogue Agent response
    socratic_response = state.get("socratic_response", "")
    if socratic_response:
        state["draft_answer"] = socratic_response
        state["tool_results"] = []
        logger.info("Using Socratic Dialogue Agent response.")
        _log_trace(step="Supervisor", purpose="Pass socratic_response as draft.", decision="socratic_response → draft_answer.")
        return state

    # 4. Fallback
    docs = state.get("retrieved_docs", [])
    context = "\n".join(d["text"] for d in docs) if docs else "No lecture materials found."
    fallback_prompt = f"You are SocrAItes. Generate a brief Socratic hint and question in Korean.\nContext: {context[:500]}\nStudent: {last_msg}"
    state["draft_answer"] = _get_content(llm.invoke([HumanMessage(content=fallback_prompt)]))
    state["tool_results"] = []
    logger.info("Fallback Supervisor generation used.")
    _log_trace(step="Supervisor", purpose="Fallback generation.", response=state["draft_answer"])
    return state
