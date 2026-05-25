"""Core Agent — Evaluator: 4-axis quality check with retry trigger.
Also contains direct_response for casual (non-study) queries.
"""
import os
import json
import logging
from typing import Any, List, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.llm import llm, _get_content
from src.agent.helpers import _log_trace

logger = logging.getLogger("SocrAItes.Agent")

EVALUATOR_PROMPT = """You are the Evaluator for SocrAItes. Evaluate the draft response on 4 axes.

Draft Response:
"{draft}"

Lecture context (excerpt):
"{context_excerpt}"

Axes (score 1-5 each):
1. socratic: Avoids direct answers, uses Socratic questioning (5=pure Socratic, 1=fully direct)
2. grounding: Grounded in provided lecture materials (5=clearly references lecture, 1=no relation)
3. encouragement: Warm and supportive tone (5=very encouraging, 1=cold)
4. clarity: Clear and well-structured (5=very clear, 1=confusing)

Pass criteria: ALL scores >= 3.
If any score < 3, set pass=false and provide a specific, actionable improvement suggestion in Korean.

Respond ONLY in JSON:
{{
  "scores": {{"socratic": <int>, "grounding": <int>, "encouragement": <int>, "clarity": <int>}},
  "pass": <bool>,
  "feedback": "<Korean suggestion, or empty string if pass>"
}}"""


def evaluator(state: AgentState) -> AgentState:
    """4-axis quality check. Increments retry_count on failure."""
    logger.info("--- [Evaluator] Quality Check Step ---")
    draft = state.get("draft_answer", "")
    docs = state.get("retrieved_docs", [])
    context_excerpt = " ".join(d["text"] for d in docs[:2])[:400] if docs else ""

    prompt = EVALUATOR_PROMPT.format(draft=draft, context_excerpt=context_excerpt)

    try:
        if os.getenv("OPENAI_API_KEY"):
            response = llm.bind(response_format={"type": "json_object"}).invoke([HumanMessage(content=prompt)])
        else:
            response = llm.invoke([HumanMessage(content=prompt)])
        parsed = json.loads(_get_content(response).strip())
        scores = parsed.get("scores", {})
        passed = parsed.get("pass", True)
        feedback = parsed.get("feedback", "")
    except Exception as e:
        logger.warning(f"Evaluator parse failed: {e}. Defaulting to pass.")
        scores, passed, feedback = {}, True, ""

    state["evaluation"] = {"pass": passed, "feedback": feedback, "scores": scores}
    if not passed:
        state["retry_count"] = state.get("retry_count", 0) + 1

    logger.info(f"Evaluation: pass={passed} scores={scores} retry={state.get('retry_count', 0)}")
    _log_trace(
        step="Evaluator",
        purpose="4-axis quality check: Socratic, grounding, encouragement, clarity.",
        inputs={"Draft length": len(draft), "Docs": len(docs)},
        prompt_details=prompt,
        response=state["evaluation"],
        decision=f"Pass={passed} | Retries={state.get('retry_count', 0)}",
    )
    return state


def direct_response(state: AgentState) -> AgentState:
    """Handles casual (non-study) queries with a brief friendly reply."""
    logger.info("--- [Direct Response] Step ---")
    last_msg = state["messages"][-1]["content"]
    system_msg = "You are a friendly academic assistant. Respond to the user's greeting or casual talk briefly in Korean."
    state["draft_answer"] = _get_content(
        llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=last_msg)])
    )
    _log_trace(
        step="DirectResponse",
        purpose="Brief Korean reply to casual/non-study message.",
        inputs={"Input": last_msg},
        response=state["draft_answer"],
    )
    return state
