"""Core Agent — Planner: selects sub-agents and defines subtask."""
import os
import json
import logging
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage

from src.agent.state import AgentState
from src.agent.llm import llm, _get_content
from src.agent.helpers import _log_trace

logger = logging.getLogger("SocrAItes.Agent")

# Keywords that always force "diagnosis" into sub_agents regardless of LLM decision
_DIAGNOSIS_KEYWORDS = [
    "퀴즈", "문제", "문제 내", "문제내", "quiz", "연습문제",           # quiz
    "약점", "약점 저장", "모르는 거 저장", "weakness",                 # weakness
    "복습 일정", "일정 등록", "schedule",                              # schedule
    "그냥 답", "답 알려줘", "답을 알려", "그냥 알려줘", "escape",       # escape
]

PLANNER_PROMPT = """You are the Planner for SocrAItes, a Socratic learning assistant.

Given the user query and context, decide which sub-agents to invoke and define the subtask.

Sub-agent selection rules:
- "retrieval": Include when the query requires looking up lecture materials (default for learning queries)
- "dialogue": Include when Socratic dialogue/questioning is needed (default for learning queries)
- "diagnosis": Include when the user explicitly requests a quiz, weakness saving, schedule, or escape to direct answer, OR when the conversation shows repeated misunderstanding

NOTE: "diagnosis" means calling a learning TOOL (generate_quiz, save_weakness, schedule_review, escape_to_answer).
Do NOT confuse quiz tool calls with Socratic dialogue.

User query: "{last_query}"
Socratic depth mode: {depth_mode}
Frustration level: {frustration_level} (0=none, higher=more frustrated)
Evaluator feedback (if retry): "{eval_feedback}"

Respond ONLY in JSON. Write "subtask" in Korean:
{{
  "sub_agents": ["retrieval", "dialogue"],
  "subtask": "이번 턴에 수행할 작업에 대한 간결한 한국어 설명",
  "reasoning": "why these sub-agents were selected"
}}"""


def _needs_diagnosis(text: str) -> bool:
    """Rule-based check: returns True if the text contains a diagnosis keyword."""
    clean = text.replace(" ", "").lower()
    return any(k.replace(" ", "").lower() in clean for k in _DIAGNOSIS_KEYWORDS)


def planner(state: AgentState) -> AgentState:
    """Decides which sub-agents to invoke and sets the execution plan."""
    logger.info("--- [Planner] Step ---")
    last_query = state.get("contextualized_query", "") or (
        state["messages"][-1]["content"] if state.get("messages") else ""
    )
    # Also check the original user message in case contextualization changed wording
    last_user_msg = state.get("messages", [{}])[-1].get("content", "")

    depth_modes = ["Light (1-2 turns)", "Standard (3-4 turns)", "Deep (5+ turns)"]
    depth_mode = depth_modes[state.get("socratic_depth", 1)]
    frustration_level = state.get("frustration_level", 0)
    eval_feedback = state.get("evaluation", {}).get("feedback", "")

    prompt = PLANNER_PROMPT.format(
        last_query=last_query,
        depth_mode=depth_mode,
        frustration_level=frustration_level,
        eval_feedback=eval_feedback,
    )

    if os.getenv("OPENAI_API_KEY"):
        response = llm.bind(response_format={"type": "json_object"}).invoke([HumanMessage(content=prompt)])
    else:
        response = llm.invoke([HumanMessage(content=prompt)])

    response_str = _get_content(response).strip()
    try:
        parsed = json.loads(response_str)
        sub_agents = parsed.get("sub_agents", ["retrieval", "dialogue"])
        subtask = parsed.get("subtask", "")
    except Exception:
        sub_agents = ["retrieval", "dialogue"]
        subtask = last_query

    # Rule-based override: force "diagnosis" if keyword detected, even if LLM missed it
    if _needs_diagnosis(last_user_msg) or _needs_diagnosis(last_query):
        if "diagnosis" not in sub_agents:
            sub_agents = sub_agents + ["diagnosis"]
            logger.info("Diagnosis keyword detected — forcing 'diagnosis' into sub_agents.")

    state["sub_agents"] = sub_agents
    state["plan"] = subtask
    state["next_step"] = "subagents"

    logger.info(f"Sub-agents: {sub_agents} | Subtask: {subtask}")
    _log_trace(
        step="Planner",
        purpose="Decide which sub-agents to invoke and define the subtask.",
        inputs={"Query": last_query, "Depth": depth_mode, "Frustration": frustration_level},
        prompt_details=prompt,
        response=response_str,
        decision={"sub_agents": sub_agents, "subtask": subtask},
    )
    return state
