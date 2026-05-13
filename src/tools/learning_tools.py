# src/tools/learning_tools.py
"""Function calling tools for SocrAItes.

The actual implementations will be wired into LangChain function calling.
For now we provide simple stubs that log the call and return a dummy
result. Replace with real logic (quiz generation, scheduling, etc.) when
the rest of the system is ready.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

from pydantic import BaseModel, Field

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

class WeaknessRecord(BaseModel):
    """Parameters for ``save_weakness`` tool.

    * ``concept`` – concept that the user struggled with.
    * ``details`` – free‑form details about the difficulty.
    """

    concept: str = Field(..., description="Concept the user is weak on")
    details: str = Field(..., description="Additional details about the weakness")

class EscapeResponse(BaseModel):
    """Parameters for ``escape_to_answer`` tool.

    * ``question`` – the original user question that triggered the escape.
    * ``answer`` – the direct answer to provide.
    """

    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Direct answer to give")

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
    # In a real system we would write to a calendar or DB.
    return {"status": "scheduled", "when": req.datetime, "description": req.description}


def save_weakness(request: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a weakness record.

    Returns an acknowledgement.
    """
    req = WeaknessRecord(**request)
    logger.info("save_weakness called with %s", req)
    # Placeholder – actual persistence handled by ``src.db`` later.
    return {"status": "saved", "concept": req.concept}


def escape_to_answer(request: Dict[str, Any]) -> Dict[str, Any]:
    """Immediately provide a direct answer, bypassing Socratic flow.
    """
    req = EscapeResponse(**request)
    logger.info("escape_to_answer called with %s", req)
    return {"answer": req.answer}

# Export a mapping for LangChain function calling registration.
TOOL_MAP = {
    "generate_quiz": generate_quiz,
    "schedule_review": schedule_review,
    "save_weakness": save_weakness,
    "escape_to_answer": escape_to_answer,
}
