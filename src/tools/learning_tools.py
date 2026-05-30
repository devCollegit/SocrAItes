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
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.db.database import (
    init_db,
    add_schedule,
    get_pending_schedules,
    save_weakness as db_save_weakness,
)
from src.agent.helpers import _log_tool_trace

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def _parse_review_datetime(value: str) -> datetime:
    """Parse user/tool provided datetime and normalize to UTC.

    - Accepts ISO-8601 with optional trailing ``Z``.
    - Naive datetime is treated as UTC.
    """
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _suggest_default_review_datetime(now_utc: datetime) -> datetime:
    """Suggest a default review datetime spread across the next 2 weeks.

    Priority offsets are 7, 10, 14 days to avoid piling all auto-schedules on one day.
    If all are occupied, fallback to 7 + N days.
    """
    used_dates: set[str] = set()
    try:
        for item in get_pending_schedules():
            raw = item.get("review_at")
            if not raw:
                continue
            dt = _parse_review_datetime(str(raw))
            if now_utc <= dt <= (now_utc + timedelta(days=21)):
                used_dates.add(dt.date().isoformat())
    except Exception:
        # If DB lookup fails, fallback safely to +7 days.
        return now_utc + timedelta(days=7)

    # Keep the original time-of-day to maintain UTC consistency.
    for offset in (7, 10, 14):
        candidate = now_utc + timedelta(days=offset)
        if candidate.date().isoformat() not in used_dates:
            return candidate

    # Fallback: find the next free date starting from +7d.
    for offset in range(7, 29):
        candidate = now_utc + timedelta(days=offset)
        if candidate.date().isoformat() not in used_dates:
            return candidate

    return now_utc + timedelta(days=29)

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
    severity: int | None = Field(None, ge=1, le=5, description="Weakness severity (1-5). If omitted, auto-estimated.")
    session_id: str | None = Field(None, description="Optional session ID")
    user_id: str | None = Field("default", description="Optional user ID")


class UserProfileRequest(BaseModel):
    """Parameters for ``update_user_profile`` tool."""
    learning_style: str | None = Field(None, description="User's preferred learning style: 'conceptual' (이론적), 'practical' (실전/사례), 'concise' (간결)")
    preferred_tone: str | None = Field(None, description="User's preferred tone: 'encouraging' (격려형), 'strict' (엄격형), 'academic' (학구형)")
    academic_background: str | None = Field(None, description="User's academic background or major")
    notes: str | None = Field(None, description="Summarized notes/observations about the user's behavior or preferences")
    strengths_summary: str | None = Field(None, description="A 2-3 sentence paragraph summarizing key active strengths/mastered concepts of the user.")
    weaknesses_summary: str | None = Field(None, description="A 2-3 sentence paragraph summarizing active weaknesses/struggles/misconceptions of the user.")
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
1. Each question MUST have exactly 4 options.
2. Options should be provided WITHOUT letter prefixes (no "A.", "B.", etc. - just the content).
3. Exactly ONE option must be the correct answer; the other three must be plausible distractors.
4. Vary question difficulty (concept recall, applied reasoning, edge cases).
5. All text must be in Korean.
6. Do NOT add any explanation text outside the JSON block.

Return ONLY a valid JSON array with this schema (no markdown fences):
[
  {{
    "question": "질문 내용",
    "options": ["선택지 내용 (문자 없음)", "선택지 내용 (문자 없음)", "선택지 내용 (문자 없음)", "선택지 내용 (문자 없음)"],
    "answer": "A"
  }},
  ...
]
"""


def _parse_quiz_json(raw: str) -> list | None:
    """Extract and parse a JSON array from raw LLM output.
    
    Also cleans up option text by removing letter prefixes (A., B., etc.)
    Normalises the ``answer`` field to a single letter (A/B/C/D).
    """
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # Find the first [...] block
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        items = json.loads(match.group(0))
        if isinstance(items, list) and items:
            letter_map = {chr(65 + i): chr(65 + i) for i in range(4)}  # A-D
            for item in items:
                # Clean up options: remove "A.", "B.", "C.", "D." prefixes if present
                if "options" in item and isinstance(item["options"], list):
                    cleaned_options = []
                    for opt in item["options"]:
                        opt_str = str(opt).strip()
                        cleaned = re.sub(r"^[A-D]\.\s*", "", opt_str)
                        cleaned_options.append(cleaned)
                    item["options"] = cleaned_options

                # Normalise answer to A/B/C/D
                raw_answer = str(item.get("answer", "")).strip()
                if raw_answer.upper() in letter_map:
                    # Already a single letter — just uppercase it
                    item["answer"] = raw_answer.upper()
                elif item.get("options"):
                    # LLM returned the full option text — find its index
                    opts = item["options"]
                    raw_answer_stripped = re.sub(r"^[A-D]\.\s*", "", raw_answer).strip()
                    matched_idx = None
                    for idx, opt in enumerate(opts):
                        if opt.strip() == raw_answer_stripped or opt.strip() == raw_answer:
                            matched_idx = idx
                            break
                    if matched_idx is not None and matched_idx < 4:
                        item["answer"] = chr(65 + matched_idx)
                    else:
                        # Fallback: keep as-is (grade_quiz will handle it)
                        item["answer"] = raw_answer
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

    res = {
        "quiz": quiz_items,
        "topic": req.topic,
        "source": "llm_rag" if context and quiz_items and os.getenv("OPENAI_API_KEY") else "template",
    }
    _log_tool_trace("generate_quiz", request, res)
    return res


def schedule_review(request: Dict[str, Any]) -> Dict[str, Any]:
    """Schedule a review session.

    Returns a confirmation with the scheduled datetime.
    """
    req = ScheduleRequest(**request)
    logger.info("schedule_review called with %s", req)
    now_utc = _utc_now()

    if req.datetime:
        try:
            review_at = _parse_review_datetime(req.datetime)
        except Exception:
            logger.warning("Invalid schedule datetime '%s'; defaulting to +7 days UTC", req.datetime)
            review_at = _suggest_default_review_datetime(now_utc)

        # Guardrail: if model/user supplied a past datetime, schedule from current UTC.
        if review_at < (now_utc - timedelta(minutes=5)):
            logger.warning(
                "Past schedule datetime '%s' detected; auto-adjusting to +7 days from current UTC",
                req.datetime,
            )
            review_at = _suggest_default_review_datetime(now_utc)
    else:
        review_at = _suggest_default_review_datetime(now_utc)

    init_db()
    schedule_id = add_schedule(
        review_at=review_at.isoformat(),
        description=req.description,
        weakness_id=req.weakness_id,
    )
    res = {
        "status": "scheduled",
        "schedule_id": schedule_id,
        "when": review_at.isoformat(),
        "description": req.description,
        "weakness_id": req.weakness_id,
    }
    _log_tool_trace("schedule_review", request, res)
    return res


def save_weakness(request: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a weakness record.

    Returns an acknowledgement.
    """
    def _estimate_weakness_severity(
        concept: str,
        details: str,
        session_id: str | None,
        user_id: str,
    ) -> int:
        """Estimate weakness severity in [1, 5] from context signals.

        Signals:
        - linguistic difficulty cues (details/concept)
        - quiz score mention in text (e.g., "40점")
        - repeated unresolved concept count
        - recent frustration cues in session messages
        """
        score = 2.0
        text = f"{concept} {details}".lower()

        severe_cues = [
            "전혀", "완전히", "아예", "너무 어렵", "계속 틀", "반복 오답", "모르겠", "이해가 안",
        ]
        moderate_cues = [
            "헷갈", "어렵", "불확실", "약함", "혼동", "정리가 안", "기억이 안",
        ]
        mild_cues = [
            "조금", "약간", "부분적으로", "살짝", "대체로 이해",
        ]

        if any(k in text for k in severe_cues):
            score += 1.5
        if any(k in text for k in moderate_cues):
            score += 0.8
        if any(k in text for k in mild_cues):
            score -= 0.5

        quiz_scores = [int(m) for m in re.findall(r"(\d{1,3})\s*점", text)]
        if quiz_scores:
            min_score = min(quiz_scores)
            if min_score <= 40:
                score += 1.5
            elif min_score <= 60:
                score += 1.0
            elif min_score >= 80:
                score -= 0.5

        try:
            from src.db.database import get_weaknesses, get_messages

            unresolved = get_weaknesses(resolved=False, limit=300, user_id=user_id)
            norm_concept = concept.strip().lower().replace(" ", "")
            repeats = sum(
                1
                for w in unresolved
                if str(w.get("concept", "")).strip().lower().replace(" ", "") == norm_concept
            )
            if repeats >= 2:
                score += 0.8
            if repeats >= 4:
                score += 0.7

            if session_id:
                recent_msgs = get_messages(session_id, limit=16)
                frustration_cues = ["모르겠", "이해가 안", "어렵", "헷갈", "답답", "막힘"]
                frustration_hits = 0
                for msg in recent_msgs:
                    if msg.get("role") != "user":
                        continue
                    c = str(msg.get("content", "")).lower()
                    if any(k in c for k in frustration_cues):
                        frustration_hits += 1
                if frustration_hits >= 2:
                    score += 0.7
                if frustration_hits >= 4:
                    score += 0.5
        except Exception as e:
            logger.warning("severity estimation context lookup failed: %s", e)

        return max(1, min(5, int(round(score))))

    req = WeaknessRecord(**request)
    logger.info("save_weakness called with %s", req)

    user_id = req.user_id or "default"
    severity_provided = isinstance(request, dict) and request.get("severity") is not None
    severity = req.severity if severity_provided and req.severity is not None else _estimate_weakness_severity(
        concept=req.concept,
        details=req.details,
        session_id=req.session_id,
        user_id=user_id,
    )

    init_db()
    weakness_id = db_save_weakness(
        concept=req.concept,
        details=req.details,
        severity=severity,
        session_id=req.session_id,
        user_id=user_id,
    )
    res = {
        "status": "saved",
        "weakness_id": weakness_id,
        "concept": req.concept,
        "severity": severity,
    }
    _log_tool_trace("save_weakness", request, res)
    return res



def grade_quiz(quiz_items: List[Dict[str, Any]], user_answers: Dict[int, str]) -> Dict[str, Any]:
    """Auto-grade a quiz.
    
    Args:
        quiz_items: List of quiz questions with 'answer' field (e.g. [{"question": "...", "answer": "A"}, ...])
        user_answers: Dict mapping question index to user's answer letter (e.g. {0: "A", 1: "B", ...})
    
    Returns:
        Dict with score, feedback, and results.
    """
    if not quiz_items:
        return {"score": 0, "correct": 0, "total": 0, "message": "퀴즈 데이터 오류"}
    
    correct = 0
    feedback_details = []
    
    for i, quiz_item in enumerate(quiz_items):
        raw_correct = str(quiz_item.get("answer", "")).strip()
        options = quiz_item.get("options", [])

        # Normalise answer to A/B/C/D if LLM returned full option text
        if raw_correct.upper() in {"A", "B", "C", "D"}:
            correct_answer = raw_correct.upper()
        else:
            # Search for matching option text
            raw_stripped = re.sub(r"^[A-D]\.\s*", "", raw_correct).strip()
            correct_answer = raw_correct.upper()  # fallback
            for idx, opt in enumerate(options):
                if str(opt).strip() == raw_stripped or str(opt).strip() == raw_correct:
                    correct_answer = chr(65 + idx)
                    break

        user_answer = str(user_answers.get(i, "")).upper() if i in user_answers else ""

        is_correct = user_answer == correct_answer
        if is_correct:
            correct += 1
            feedback_details.append(f"✅ {i+1}번: 정답")
        else:
            feedback_details.append(f"❌ {i+1}번: 오답 (정답: {correct_answer})")
        
        logger.info("Quiz Q%d: correct=%s, user=%s, match=%s", i+1, correct_answer, user_answer, is_correct)
    
    score = (correct / len(quiz_items)) * 100
    message = f"✅ {correct}/{len(quiz_items)} 정답입니다! 점수: {int(score)}점\n" + "\n".join(feedback_details)
    
    res = {
        "score": int(score),
        "correct": correct,
        "total": len(quiz_items),
        "message": message,
        "details": feedback_details,
        "suggest_weakness": score < 60,  # 60점 미만이면 약점 제안
    }
    
    logger.info("Quiz grading: score=%d%%, correct=%d/%d, suggest_weakness=%s", score, correct, len(quiz_items), score < 60)
    return res


def _tool_generate_quiz(topic: str, num_questions: int = 5) -> Dict[str, Any]:
    return generate_quiz({"topic": topic, "num_questions": num_questions})


def _tool_schedule_review(datetime: str | None = None, description: str | None = None, weakness_id: int | None = None) -> Dict[str, Any]:
    return schedule_review({"datetime": datetime, "description": description, "weakness_id": weakness_id})


def _tool_save_weakness(
    concept: str,
    details: str,
    severity: int | None = None,
    session_id: str | None = None,
    user_id: str | None = "default",
) -> Dict[str, Any]:
    return save_weakness({"concept": concept, "details": details, "severity": severity, "session_id": session_id, "user_id": user_id})



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
        strengths_summary=req.strengths_summary,
        weaknesses_summary=req.weaknesses_summary,
    )
    res = {
        "status": "updated",
        "user_id": user_id,
        "profile": {
            "learning_style": req.learning_style,
            "preferred_tone": req.preferred_tone,
            "academic_background": req.academic_background,
            "notes": req.notes,
        }
    }
    _log_tool_trace("update_user_profile", request, res)
    return res


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
    res = {
        "status": "saved",
        "strength_id": strength_id,
        "concept": req.concept,
    }
    _log_tool_trace("save_strength", request, res)
    return res


def _tool_update_user_profile(
    learning_style: str | None = None,
    preferred_tone: str | None = None,
    academic_background: str | None = None,
    notes: str | None = None,
    strengths_summary: str | None = None,
    weaknesses_summary: str | None = None,
    user_id: str = "default",
) -> Dict[str, Any]:
    return update_user_profile({
        "learning_style": learning_style,
        "preferred_tone": preferred_tone,
        "academic_background": academic_background,
        "notes": notes,
        "strengths_summary": strengths_summary,
        "weaknesses_summary": weaknesses_summary,
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

# ---------------------------------------------------------------------------
# Web Search Tool (Deep 모드 전용 — Tavily)
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., description="검색할 질문 또는 키워드")
    max_results: int = Field(3, ge=1, le=5, description="반환할 검색 결과 수")


def search_web(request: Dict[str, Any]) -> Dict[str, Any]:
    """Tavily를 이용해 웹 검색 후 결과를 반환한다. Deep 모드 전용."""
    req = SearchRequest(**request)
    logger.info("search_web called: query=%s", req.query)

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"status": "error", "message": "TAVILY_API_KEY가 설정되지 않았습니다."}

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=req.query,
            max_results=req.max_results,
            search_depth="advanced",
        )
        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
            for r in response.get("results", [])
        ]
        res = {"status": "success", "query": req.query, "results": results}
    except Exception as e:
        logger.warning("Tavily search failed: %s", e)
        res = {"status": "error", "message": str(e)}

    _log_tool_trace("search_web", request, res)
    return res


def _tool_search_web(query: str, max_results: int = 3) -> Dict[str, Any]:
    return search_web({"query": query, "max_results": max_results})


# Export a mapping for LangChain function calling registration.
TOOL_MAP = {
    "generate_quiz": generate_quiz,
    "schedule_review": schedule_review,
    "save_weakness": save_weakness,
    "update_user_profile": update_user_profile,
    "save_strength": save_strength,
    "search_web": search_web,
}


_SEARCH_WEB_TOOL = StructuredTool.from_function(
    name="search_web",
    description="Deep 모드 전용: 강의자료에 없는 최신 정보나 추가 맥락이 필요할 때 웹 검색을 수행한다.",
    func=_tool_search_web,
)

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

# Deep 모드 전용 툴 목록 (search_web 포함)
LANGCHAIN_TOOLS_DEEP: List[StructuredTool] = LANGCHAIN_TOOLS + [_SEARCH_WEB_TOOL]
