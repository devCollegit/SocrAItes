"""Sub Agent — Socratic Dialogue: generates Socratic questions and hints."""
import logging
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.llm import llm, _get_content
from src.agent.helpers import _log_trace, _is_summary_request

logger = logging.getLogger("SocrAItes.Agent")

SOCRATIC_DIALOGUE_PROMPT = """You are SocrAItes, a world-class Socratic tutor.

Your task: Generate a Socratic response in Korean that guides the student to think critically.
- Do NOT give direct answers. Provide a brief hint or high-level context (1-2 sentences), then ask a concrete, thought-provoking Socratic question.
- Exception: If the student requested a summary, provide a comprehensive structured summary followed by a Socratic question.
- If frustration_level >= 2, offer more scaffolding (a more detailed hint) before asking the question.
- Ask about specific sub-components, mechanisms, or key distinctions — never vague opinion questions.

Progressive Learning & Context Progression (MANDATORY):
- Carefully analyze the conversation history. If a sub-concept (e.g., stemming vs lemmatization) has already been discussed or explained by the student (normally 2-3 turns or once they've correctly stated the definition/difference), DO NOT keep drilling on that same topic.
- Instead, actively bridge to the next logical concept present in the Lecture Context (e.g., stop words removal, normalization, tokenization, or the next phase of parsing).
- Avoid loop questions. If the student answers your question reasonably well, acknowledge their correct understanding briefly (with positive reinforcement), and then introduce the next topic or challenge.

Personalization Guidelines:
1. Learning Style: If '{learning_style}' is 'practical' (실전/사례), use real-world software examples and case studies. If 'conceptual' (이론적), focus on theoretical and mathematical foundations. If 'concise' (간결), keep explanations extremely brief.
2. Preferred Tone: If '{preferred_tone}' is 'encouraging' (격려형), use a warm, empathetic tone with compliments. If 'strict' (엄격형), focus strictly on factual details and challenge assumptions without soft padding. If 'academic' (학구형), use formal graduate-level research vocabulary.
3. Academic Background: The student is a '{academic_background}'. Tailor your explanations and analogies to match their background knowledge.
4. AI Notes on Student: {profile_notes}
5. Leverage Strengths: If possible, draw analogies using concepts the student is strong in:
{strengths_text}
6. Address Weaknesses: Pay extra attention to concepts the student has struggled with in the past:
{weaknesses_text}

Socratic depth: {depth} (0=Light 1-2 turns, 1=Standard 3-4 turns, 2=Deep 5+ turns)
Frustration level: {frustration_level}
Evaluator feedback (improve if non-empty): "{eval_feedback}"

Lecture context:
---
{context}
---

Conversation history:
{history}"""


def socratic_dialogue_agent(state: AgentState) -> AgentState:
    """Generates Socratic questions/hints when 'dialogue' is in sub_agents."""
    logger.info("--- [Socratic Dialogue Agent] Step ---")
    if "dialogue" not in state.get("sub_agents", ["dialogue"]):
        state["socratic_response"] = ""
        _log_trace(step="SocraticDialogueAgent", purpose="Skipped.", decision="Not in sub_agents.")
        return state

    docs = state.get("retrieved_docs", [])
    history = state.get("messages", [])
    depth = state.get("socratic_depth", 1)
    frustration_level = state.get("frustration_level", 0)
    eval_feedback = state.get("evaluation", {}).get("feedback", "")

    session_id = state.get("session_id")
    user_id = "default"
    if session_id:
        try:
            from src.db.database import get_session
            sess = get_session(session_id)
            if sess:
                user_id = sess.get("user_id", "default")
        except Exception as e:
            logger.warning(f"Failed to fetch session details: {e}")

    # Fetch user unresolved weaknesses
    weaknesses_text = "No unresolved weaknesses recorded yet."
    try:
        from src.db.database import get_user_unresolved_weaknesses
        unresolved_weaknesses = get_user_unresolved_weaknesses(user_id, limit=5)
        if unresolved_weaknesses:
            weaknesses_text = "\n".join(
                f"- {w['concept']}: {w['details']} (severity: {w['severity']})"
                for w in unresolved_weaknesses
            )
    except Exception as e:
        logger.warning(f"Failed to fetch user weaknesses: {e}")

    # Fetch user strengths
    strengths_text = "No recorded strengths yet."
    try:
        from src.db.database import get_user_strengths
        user_strengths = get_user_strengths(user_id, limit=5)
        if user_strengths:
            strengths_text = "\n".join(
                f"- {s['concept']}: {s['details']}"
                for s in user_strengths
            )
    except Exception as e:
        logger.warning(f"Failed to fetch user strengths: {e}")

    # Load user profile preferences
    user_profile = state.get("user_profile", {})
    learning_style = user_profile.get("learning_style", "conceptual")
    preferred_tone = user_profile.get("preferred_tone", "encouraging")
    academic_background = user_profile.get("academic_background", "대학원생")
    profile_notes = user_profile.get("notes", "None")

    context = "\n".join(d["text"] for d in docs) if docs else "No lecture materials found."
    last_msg = history[-1]["content"] if history else ""

    is_summary = _is_summary_request(last_msg) or _is_summary_request(state.get("contextualized_query", ""))
    summary_override = (
        "IMPORTANT: The student requested a summary. "
        "Provide a comprehensive, structured, detailed summary of the lecture context first, "
        "then close with one concrete Socratic question.\n"
        if is_summary else ""
    )

    history_text = "\n".join(
        f"{'Student' if m['role'] == 'user' else 'Tutor'}: {m['content']}"
        for m in history
    )

    system_content = summary_override + SOCRATIC_DIALOGUE_PROMPT.format(
        depth=depth,
        frustration_level=frustration_level,
        eval_feedback=eval_feedback,
        learning_style=learning_style,
        preferred_tone=preferred_tone,
        academic_background=academic_background,
        profile_notes=profile_notes,
        strengths_text=strengths_text,
        weaknesses_text=weaknesses_text,
        context=context,
        history=history_text,
    )

    response = llm.invoke([SystemMessage(content=system_content), HumanMessage(content=last_msg)])
    socratic_response = _get_content(response)
    state["socratic_response"] = socratic_response

    logger.info(f"Socratic response generated (len={len(socratic_response)}).")
    _log_trace(
        step="SocraticDialogueAgent",
        purpose="Generate Socratic questions/hints from retrieved lecture content.",
        inputs={"Depth": depth, "Frustration": frustration_level, "Docs": len(docs)},
        prompt_details=system_content,
        response=socratic_response,
        decision="socratic_response saved.",
    )
    return state
