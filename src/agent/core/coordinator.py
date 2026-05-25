"""Core Agent — Coordinator: query reformulation and routing."""
import os
import re
import json
import logging
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage

from src.agent.state import AgentState
from src.agent.llm import llm, _get_content
from src.agent.helpers import _log_trace, _detect_frustration, _is_last_message_quiz_prompt, _format_history

logger = logging.getLogger("SocrAItes.Agent")

COMBINED_COORDINATOR_PROMPT = """You are the Coordinator and Query Reformulator for SocrAItes, a Socratic learning assistant.

Your task is to analyze the conversation history and the latest user message to do two things:
1. Reformulate the latest user message into a standalone, search-optimized query in Korean.
2. Classify whether the user's intent is study-related (academic concepts, lecture materials) or a casual interaction (greetings, gratitude, off-topic, simple navigation).

Instructions for Query Reformulation:
1. Identify the core academic concept or question the user is asking about.
2. Incorporate necessary context (concepts, terms, definitions) from the previous turns so the query is fully self-contained.
3. Keep the query concise, focused on key concepts, and ideal for retrieval from lecture materials (PDF).
4. If the latest message is a simple greeting, thank you, or casual chit-chat, return it exactly as-is.
5. If the latest message asks to continue or summarize "again" (e.g., "다시 요약해줘", "다음 내용 알려줘"), reformulate to search for the NEXT sequential topics rather than repeating what was already explained.

Instructions for Classification:
- Output 'PLAN' if the query is learning/study-related (lecture materials, academic concepts, quizzes).
- Output 'DIRECT' if the query is casual (greeting, thanks, off-topic).

Respond ONLY in JSON:
{{"contextualized_query": "<Korean query>", "routing_decision": "PLAN" | "DIRECT"}}

Conversation History:
{history_text}

Latest User Message:
{last_message}"""


def coordinator(state: AgentState) -> AgentState:
    """Reformulates follow-up queries and routes to planner or direct_response."""
    logger.info("--- [Coordinator] Step ---")
    messages = state.get("messages", [])
    last_message = messages[-1]["content"] if messages else ""

    frustration_count = sum(
        1 for msg in messages
        if msg.get("role") == "user" and _detect_frustration(msg.get("content", ""))
    )
    state["frustration_level"] = frustration_count

    pending_quiz = state.get("pending_quiz", [])
    is_quiz_pending = len(pending_quiz) > 0 or _is_last_message_quiz_prompt(messages[:-1])
    is_frustrated_in_quiz = (
        is_quiz_pending
        and messages
        and messages[-1].get("role") == "user"
        and _detect_frustration(messages[-1].get("content", ""))
    )

    # Bypass LLM if frustrated during a quiz — route directly to planner
    if is_frustrated_in_quiz:
        state["contextualized_query"] = last_message
        state["next_step"] = "planner"
        logger.info("Frustrated during quiz — bypassing LLM, routing to planner.")
        _log_trace(
            step="Coordinator",
            purpose="Bypassed LLM due to frustration during quiz.",
            inputs={"Message": last_message, "Frustration": frustration_count},
            decision="Routing to planner with original query.",
        )
        return state

    is_first_turn = len(messages) <= 1
    history_text = _format_history(messages)
    prompt = COMBINED_COORDINATOR_PROMPT.format(history_text=history_text, last_message=last_message)

    if os.getenv("OPENAI_API_KEY"):
        response = llm.bind(response_format={"type": "json_object"}).invoke([HumanMessage(content=prompt)])
    else:
        response = llm.invoke([HumanMessage(content=prompt)])

    response_str = _get_content(response).strip()
    # Strip markdown code fences if present
    content = re.sub(r"^```json\s*", "", response_str, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content).strip()

    try:
        parsed = json.loads(content)
        rewritten = parsed.get("contextualized_query", "").strip()
        decision = parsed.get("routing_decision", "").strip().upper()
    except Exception as e:
        logger.error(f"Coordinator JSON parse failed: {e}. Falling back.")
        rewritten = last_message
        decision = "PLAN" if "PLAN" in content.upper() else "DIRECT"

    # Fallback to original query on first turn or empty result
    if is_first_turn or not rewritten:
        rewritten = last_message

    state["contextualized_query"] = rewritten
    state["next_step"] = "planner" if "PLAN" in decision else "direct_response"

    logger.info(f"Query: '{rewritten}' | Route: {state['next_step']}")
    _log_trace(
        step="Coordinator",
        purpose="Reformulate query and classify intent (PLAN vs DIRECT).",
        inputs={"Original": last_message, "History turns": len(messages) - 1, "Frustration": frustration_count},
        prompt_details={"History": history_text, "Prompt": prompt},
        response=response_str,
        decision={"Contextualized Query": rewritten, "Routing": state["next_step"]},
    )
    return state
