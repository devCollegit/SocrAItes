# src/tools/learning_tools.py
"""Function calling tools for SocrAItes.

The actual implementations will be wired into LangChain function calling.
For now we provide simple stubs that log the call and return a dummy
result. Replace with real logic (quiz generation, scheduling, etc.) when
the rest of the system is ready.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Any, List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.db.database import init_db, add_schedule, save_weakness as db_save_weakness

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schemas – these define the JSON schema exposed to the LLM.
# ---------------------------------------------------------------------------

class QuizRequest(BaseModel):
    """Parameters for ``generate_quiz`` tool.

    * ``topic`` – the concept or keyword to base the quiz on.
    * ``num_questions`` – number of questions to generate.
    """

    topic: str = Field(..., description="Concept or keyword for the quiz")
    num_questions: int = Field(5, ge=1, le=20, description="Number of quiz questions")

class ScheduleRequest(BaseModel):
    """Parameters for ``schedule_review`` tool.

    * ``datetime`` – when the review should occur (ISO 8601 string).
    * ``description`` – optional note about the review.
    """

    datetime: str = Field(..., description="ISO 8601 datetime for the review")
    description: str | None = Field(None, description="Optional description of the review")
    weakness_id: int | None = Field(None, description="Optional weakness ID linked to this review")

class WeaknessRecord(BaseModel):
    """Parameters for ``save_weakness`` tool.

    * ``concept`` – concept that the user struggled with.
    * ``details`` – free‑form details about the difficulty.
    """

    concept: str = Field(..., description="Concept the user is weak on")
    details: str = Field(..., description="Additional details about the weakness")
    severity: int = Field(2, ge=1, le=5, description="Weakness severity (1-5)")
    session_id: str | None = Field(None, description="Optional session ID")

class EscapeResponse(BaseModel):
    """Parameters for ``escape_to_answer`` tool.

    * ``question`` – the original user question that triggered the escape.
    * ``answer`` – optional direct answer to provide.
    """

    question: str = Field(..., description="Original user question")
    answer: str | None = Field(None, description="Optional direct answer to give")

# ---------------------------------------------------------------------------
# Stub implementations – they simply log and return a placeholder value.
# ---------------------------------------------------------------------------

def generate_quiz(request: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a quiz for a given topic.

    In the MVP we return a static list of multiple‑choice questions.
    """
    req = QuizRequest(**request)
    logger.info("generate_quiz called with %s", req)
    # Dummy quiz – replace with real LLM generation later.
    quiz = [
        {"question": f"What is the definition of {req.topic}?", "options": ["A", "B", "C", "D"], "answer": "A"}
        for _ in range(req.num_questions)
    ]
    return {"quiz": quiz}


def schedule_review(request: Dict[str, Any]) -> Dict[str, Any]:
    """Schedule a review session.

    Returns a confirmation with the scheduled datetime.
    """
    req = ScheduleRequest(**request)
    logger.info("schedule_review called with %s", req)
    review_at = datetime.fromisoformat(req.datetime)
    init_db()
    schedule_id = add_schedule(
        review_at=review_at.isoformat(),
        description=req.description,
        weakness_id=req.weakness_id,
    )
    return {
        "status": "scheduled",
        "schedule_id": schedule_id,
        "when": review_at.isoformat(),
        "description": req.description,
        "weakness_id": req.weakness_id,
    }


def save_weakness(request: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a weakness record.

    Returns an acknowledgement.
    """
    req = WeaknessRecord(**request)
    logger.info("save_weakness called with %s", req)
    init_db()
    weakness_id = db_save_weakness(
        concept=req.concept,
        details=req.details,
        severity=req.severity,
        session_id=req.session_id,
    )
    return {
        "status": "saved",
        "weakness_id": weakness_id,
        "concept": req.concept,
        "severity": req.severity,
    }


def escape_to_answer(request: Dict[str, Any]) -> Dict[str, Any]:
    """Immediately provide a direct answer, bypassing Socratic flow.
    """
    req = EscapeResponse(**request)
    logger.info("escape_to_answer called with %s", req)
    return {
        "mode": "direct_answer",
        "question": req.question,
        "answer": req.answer,
        "message": "User requested direct answer mode.",
    }


def _tool_generate_quiz(topic: str, num_questions: int = 5) -> Dict[str, Any]:
    return generate_quiz({"topic": topic, "num_questions": num_questions})


def _tool_schedule_review(datetime: str, description: str | None = None, weakness_id: int | None = None) -> Dict[str, Any]:
    return schedule_review({"datetime": datetime, "description": description, "weakness_id": weakness_id})


def _tool_save_weakness(
    concept: str,
    details: str,
    severity: int = 2,
    session_id: str | None = None,
) -> Dict[str, Any]:
    return save_weakness({"concept": concept, "details": details, "severity": severity, "session_id": session_id})


def _tool_escape_to_answer(question: str, answer: str | None = None) -> Dict[str, Any]:
    return escape_to_answer({"question": question, "answer": answer})

# Export a mapping for LangChain function calling registration.
TOOL_MAP = {
    "generate_quiz": generate_quiz,
    "schedule_review": schedule_review,
    "save_weakness": save_weakness,
    "escape_to_answer": escape_to_answer,
}


LANGCHAIN_TOOLS: List[StructuredTool] = [
    StructuredTool.from_function(
        name="generate_quiz",
        description="Generate short quiz questions based on a study topic.",
        func=_tool_generate_quiz,
    ),
    StructuredTool.from_function(
        name="schedule_review",
        description="Register a review schedule in ISO 8601 datetime format.",
        func=_tool_schedule_review,
    ),
    StructuredTool.from_function(
        name="save_weakness",
        description="Save a weak concept with details and optional severity/session_id.",
        func=_tool_save_weakness,
    ),
    StructuredTool.from_function(
        name="escape_to_answer",
        description="Switch to direct-answer mode for the current question.",
        func=_tool_escape_to_answer,
    ),
]
