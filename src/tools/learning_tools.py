# src/tools/learning_tools.py
"""Function calling tools for SocrAItes.

Tools are wired into LangChain function calling via LANGCHAIN_TOOLS and
executed by the supervisor node via TOOL_MAP.
"""

from __future__ import annotations

import json
import logging
import os
import re
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
    session_id: str | None = Field(None, description="Optional session ID to track history")

class ScheduleRequest(BaseModel):
    """Parameters for ``schedule_review`` tool.

    * ``datetime`` – when the review should occur (ISO 8601 string). Defaults to 7 days from now.
    * ``description`` – optional note about the review.
    """

    datetime: str | None = Field(None, description="ISO 8601 datetime for the review. If omitted, defaults to 7 days from now.")
    description: str | None = Field(None, description="Optional description of the review")
    weakness_id: int | None = Field(None, description="Optional weakness ID linked to this review")

class WeaknessRecord(BaseModel):
    """Parameters for ``save_weakness`` tool.

    * ``concept`` – concept that the user struggled with. Infer from recent conversation if not explicit.
    * ``details`` – free‑form details about the difficulty.
    """

    concept: str = Field(..., description="Concept the user is weak on. Infer from conversation context.")
    details: str = Field("학습 중 이해가 어려운 개념으로 식별됨", description="Additional details about the weakness")
    severity: int = Field(2, ge=1, le=5, description="Weakness severity (1-5)")
    session_id: str | None = Field(None, description="Optional session ID")
    user_id: str | None = Field("default", description="Optional user ID")

class EscapeResponse(BaseModel):
    """Parameters for ``escape_to_answer`` tool.

    * ``question`` – the original user question that triggered the escape.
    * ``answer`` – optional direct answer to provide.
    """

    question: str = Field(..., description="Original user question")
    answer: str | None = Field(None, description="Optional direct answer to give")


class UserProfileRequest(BaseModel):
    """Parameters for ``update_user_profile`` tool."""
    learning_style: str | None = Field(None, description="User's preferred learning style: 'conceptual' (이론적), 'practical' (실전/사례), 'concise' (간결)")
    preferred_tone: str | None = Field(None, description="User's preferred tone: 'encouraging' (격려형), 'strict' (엄격형), 'academic' (학구형)")
    academic_background: str | None = Field(None, description="User's academic background or major")
    notes: str | None = Field(None, description="Summarized notes/observations about the user's behavior or preferences")
    user_id: str | None = Field("default", description="Optional user ID")


class StrengthRecord(BaseModel):
    """Parameters for ``save_strength`` tool."""
    concept: str = Field(..., description="Concept the user is strong on. Infer from conversation context.")
    details: str = Field("학습 중 개념 이해도가 높은 것으로 식별됨", description="Additional details about the strength")
    session_id: str | None = Field(None, description="Optional session ID")
    user_id: str | None = Field("default", description="Optional user ID")

# ---------------------------------------------------------------------------
# Stub implementations – they simply log and return a placeholder value.
# ---------------------------------------------------------------------------

def _retrieve_context_for_quiz(topic: str, k: int = 6) -> str:
    """Search ES for relevant passages to use as quiz source material.

    Falls back to an empty string if ES is unavailable.
    """
    try:
        from src.rag.vectorstore import query as vs_query
        docs = vs_query(topic, k=k)
        if docs:
            return "\n\n".join(d.get("text", "") for d in docs if d.get("text"))
    except Exception as e:
        logger.warning("RAG retrieval for quiz failed: %s", e)
    return ""


_QUIZ_GENERATION_PROMPT = """\
You are an expert exam question writer for a university-level course.

Your task: generate exactly {n} multiple-choice questions about "{topic}".

Source material (use this as the primary basis; additional knowledge is fine):
---
{context}
---

Requirements:
1. Each question MUST have exactly 4 options labeled A, B, C, D.
2. Exactly ONE option must be the correct answer; the other three must be plausible distractors.
3. Vary question difficulty (concept recall, applied reasoning, edge cases).
4. All text must be in Korean.
5. Do NOT add any explanation text outside the JSON block.

Return ONLY a valid JSON array with this schema (no markdown fences):
[
  {{
    "question": "질문 내용",
    "options": ["A 보기", "B 보기", "C 보기", "D 보기"],
    "answer": "A"
  }},
  ...
]
"""


def _parse_quiz_json(raw: str) -> list | None:
    """Extract and parse a JSON array from raw LLM output."""
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # Find the first [...] block
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        items = json.loads(match.group(0))
        if isinstance(items, list) and items:
            return items
    except json.JSONDecodeError:
        pass
    return None


# ---------------------------------------------------------------------------
# Fallback static templates (used when LLM/ES unavailable)
# ---------------------------------------------------------------------------

def _get_fallback_templates(topic: str, n: int) -> list:
    templates = [
        {
            "question": f"{topic}의 가장 적절한 정의는 무엇인가요?",
            "options": [
                "시스템 자원을 효율적으로 관리하는 핵심 개념",
                "단순한 UI 디자인 원칙",
                "네트워크 케이블 규격",
                "파일 확장자 규칙",
            ],
            "answer": "A",
        },
        {
            "question": f"{topic}를 적용할 때 가장 중요한 목표로 적절한 것은?",
            "options": [
                "자원 사용의 일관성과 안정성 확보",
                "무조건 코드 줄 수 늘리기",
                "오류를 숨겨서 실행 유지하기",
                "문서 없이 빠르게 배포하기",
            ],
            "answer": "A",
        },
        {
            "question": f"{topic}의 부재로 인해 가장 가능성이 높은 문제는?",
            "options": [
                "성능 저하 또는 예측 불가능한 동작",
                "해상도 자동 향상",
                "저장공간 무한 확장",
                "CPU 발열 완전 제거",
            ],
            "answer": "A",
        },
        {
            "question": f"다음 중 {topic} 학습에 가장 효과적인 접근은?",
            "options": [
                "개념-원리-사례를 연결해 설명해보기",
                "정의만 암기하고 예시는 생략",
                "오답 분석 없이 반복 풀이",
                "관련 용어를 무시하고 구현부터 진행",
            ],
            "answer": "A",
        },
        {
            "question": f"{topic}를 실무에 적용할 때 먼저 확인해야 할 것은?",
            "options": [
                "요구사항과 제약 조건",
                "폰트 스타일",
                "파일명 길이",
                "운영체제 배경화면",
            ],
            "answer": "A",
        },
    ]
    return [templates[i % len(templates)] for i in range(n)]


def generate_quiz(request: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a quiz for a given topic using RAG + LLM.

    1. Retrieves relevant passages from Elasticsearch.
    2. Asks the LLM to generate questions grounded in that context.
    3. Falls back to static templates if LLM/ES is unavailable.
    """
    req = QuizRequest(**request)
    logger.info("generate_quiz called: topic=%s, n=%d, session_id=%s", req.topic, req.num_questions, req.session_id)

    # 1. Fetch previous quizzes from database to avoid duplicates
    previous_quizzes_context = ""
    if req.session_id:
        try:
            from src.db.database import get_messages
            db_messages = get_messages(req.session_id, limit=30)
            prev_questions = []
            for db_msg in db_messages:
                if db_msg["role"] == "assistant":
                    content = db_msg["content"]
                    # Extract questions matching common formats like "1. [Question]"
                    questions_in_msg = re.findall(r"\b\d+\.\s*(.+)", content)
                    for q in questions_in_msg:
                        clean_q = q.strip()
                        if "답안은" not in clean_q and "예:" not in clean_q and len(clean_q) > 5:
                            prev_questions.append(clean_q)
            
            if prev_questions:
                previous_quizzes_context = "\n".join(f"- {q}" for q in prev_questions)
                logger.info("Found %d previous quiz questions in session: %s", len(prev_questions), req.session_id)
        except Exception as e:
            logger.warning("Failed to retrieve quiz history from database: %s", e)

    context = _retrieve_context_for_quiz(req.topic)
    quiz_items = None

    if os.getenv("OPENAI_API_KEY") and context:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
            gen_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            
            # Incorporate previous quiz history into the prompt
            history_prompt = ""
            if previous_quizzes_context:
                history_prompt = (
                    "\n\n이전에 이미 출제된 퀴즈 질문 목록입니다. 아래 질문들과 완전히 동일하거나 중복되거나 거의 유사한 문제는 절대로 출제하지 마십시오:\n"
                    f"{previous_quizzes_context}\n"
                )

            prompt = _QUIZ_GENERATION_PROMPT.format(
                n=req.num_questions,
                topic=req.topic,
                context=context[:4000],  # stay within token limits
            ) + history_prompt
            
            response = gen_llm.invoke([HumanMessage(content=prompt)])
            raw = response.content if hasattr(response, "content") else str(response)
            quiz_items = _parse_quiz_json(raw)
            if quiz_items:
                logger.info("LLM quiz generation succeeded: %d questions", len(quiz_items))
            else:
                logger.warning("Failed to parse LLM quiz output; falling back to templates")
        except Exception as e:
            logger.warning("LLM quiz generation error: %s", e)

    if not quiz_items:
        logger.info("Using fallback quiz templates for topic: %s", req.topic)
        quiz_items = _get_fallback_templates(req.topic, req.num_questions)

    # Normalise: ensure exactly num_questions items
    quiz_items = quiz_items[:req.num_questions]

    return {
        "quiz": quiz_items,
        "topic": req.topic,
        "source": "llm_rag" if context and quiz_items and os.getenv("OPENAI_API_KEY") else "template",
    }


def schedule_review(request: Dict[str, Any]) -> Dict[str, Any]:
    """Schedule a review session.

    Returns a confirmation with the scheduled datetime.
    """
    req = ScheduleRequest(**request)
    logger.info("schedule_review called with %s", req)
    if req.datetime:
        review_at = datetime.fromisoformat(req.datetime)
    else:
        from datetime import timedelta
        review_at = datetime.now() + timedelta(days=7)
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
        user_id=req.user_id or "default",
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


def _tool_schedule_review(datetime: str | None = None, description: str | None = None, weakness_id: int | None = None) -> Dict[str, Any]:
    return schedule_review({"datetime": datetime, "description": description, "weakness_id": weakness_id})


def _tool_save_weakness(
    concept: str,
    details: str,
    severity: int = 2,
    session_id: str | None = None,
    user_id: str | None = "default",
) -> Dict[str, Any]:
    return save_weakness({"concept": concept, "details": details, "severity": severity, "session_id": session_id, "user_id": user_id})


def _tool_escape_to_answer(question: str, answer: str | None = None) -> Dict[str, Any]:
    return escape_to_answer({"question": question, "answer": answer})


def update_user_profile(request: Dict[str, Any]) -> Dict[str, Any]:
    """Persist updates to the user profile."""
    req = UserProfileRequest(**request)
    user_id = req.user_id or "default"
    logger.info("update_user_profile called for user=%s: %s", user_id, req)
    from src.db.database import save_user_profile
    save_user_profile(
        user_id=user_id,
        learning_style=req.learning_style,
        preferred_tone=req.preferred_tone,
        academic_background=req.academic_background,
        notes=req.notes,
    )
    return {
        "status": "updated",
        "user_id": user_id,
        "profile": {
            "learning_style": req.learning_style,
            "preferred_tone": req.preferred_tone,
            "academic_background": req.academic_background,
            "notes": req.notes,
        }
    }


def save_strength(request: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a strength record."""
    req = StrengthRecord(**request)
    logger.info("save_strength called with %s", req)
    from src.db.database import save_strength as db_save_strength
    strength_id = db_save_strength(
        concept=req.concept,
        details=req.details,
        session_id=req.session_id,
        user_id=req.user_id or "default",
    )
    return {
        "status": "saved",
        "strength_id": strength_id,
        "concept": req.concept,
    }


def _tool_update_user_profile(
    learning_style: str | None = None,
    preferred_tone: str | None = None,
    academic_background: str | None = None,
    notes: str | None = None,
    user_id: str = "default",
) -> Dict[str, Any]:
    return update_user_profile({
        "learning_style": learning_style,
        "preferred_tone": preferred_tone,
        "academic_background": academic_background,
        "notes": notes,
        "user_id": user_id,
    })


def _tool_save_strength(
    concept: str,
    details: str = "학습 중 개념 이해도가 높은 것으로 식별됨",
    session_id: str | None = None,
    user_id: str | None = "default",
) -> Dict[str, Any]:
    return save_strength({
        "concept": concept,
        "details": details,
        "session_id": session_id,
        "user_id": user_id,
    })

# Export a mapping for LangChain function calling registration.
TOOL_MAP = {
    "generate_quiz": generate_quiz,
    "schedule_review": schedule_review,
    "save_weakness": save_weakness,
    "escape_to_answer": escape_to_answer,
    "update_user_profile": update_user_profile,
    "save_strength": save_strength,
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
    StructuredTool.from_function(
        name="update_user_profile",
        description="Update user's profile, preferred learning style, tone, academic background, or notes.",
        func=_tool_update_user_profile,
    ),
    StructuredTool.from_function(
        name="save_strength",
        description="Save a strong concept that the user has shown good understanding of.",
        func=_tool_save_strength,
    ),
]
