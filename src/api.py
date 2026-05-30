from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import shutil
import uuid
import os
import logging
import traceback
import json


# Load .env file BEFORE anything else (so OPENAI_API_KEY is available)
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("socraites.api")

from .agent.graph import GRAPH
from .agent.state import DEFAULT_STATE
from .agent.llm import llm
from .rag.document_processor import process_pdf, compute_file_hash
from .rag.vectorstore import add_documents, get_registered_documents, delete_document
from .db.database import (
    init_db,
    create_session,
    log_message,
    get_messages,
    list_sessions,
    delete_session,
    get_weaknesses,
    delete_weakness,
    get_pending_schedules,
    delete_schedule,
    get_user_profile,
    get_session,
    get_user_strengths,
    get_reports,
    save_report,
)

app = FastAPI(title="SocrAItes API")
init_db()

# Ensure uploads directory exists
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files
frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
if not os.path.exists(frontend_path):
    os.makedirs(frontend_path)

app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    socratic_depth: Optional[int] = 1
    session_id: Optional[str] = None
    selected_docs: Optional[List[str]] = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    retrieved_docs: List[Any] = []
    subtask: Optional[str] = None
    tool_results: List[Any] = []

@app.post("/chat")
async def chat(request: ChatRequest):
    # 세션 생성 또는 기존 세션 사용
    if request.session_id:
        session_id = request.session_id
        try:
            sess = get_session(session_id)
            user_id = sess.get("user_id", "default") if sess else "default"
        except Exception:
            user_id = "default"
    else:
        first_msg = request.messages[-1].content if request.messages else "새 대화"
        title = first_msg[:30] + ("..." if len(first_msg) > 30 else "")
        session_id = create_session(title=title)
        user_id = "default"
    logger.info(f"[/chat] session={session_id} | user_id={user_id} | messages={len(request.messages)} | depth={request.socratic_depth}")
    try:
        initial_messages = [{"role": m.role, "content": m.content} for m in request.messages]
        initial_messages = initial_messages[-20:]  # 최근 10턴(20개 메시지)만 유지
        logger.debug(f"[/chat] last user message: {initial_messages[-1]['content'][:100] if initial_messages else '(empty)'}")

        # 사용자 메시지 DB 저장
        if initial_messages:
            log_message(session_id, "user", initial_messages[-1]["content"])

        # Restore pending_quiz from database metadata if it exists
        restored_pending_quiz = []
        try:
            db_messages = get_messages(session_id, limit=20)
            # Find the most recent assistant message with pending_quiz metadata
            for db_msg in reversed(db_messages):
                if db_msg["role"] == "assistant" and db_msg.get("metadata"):
                    meta = json.loads(db_msg["metadata"])
                    if "pending_quiz" in meta and meta["pending_quiz"]:
                        restored_pending_quiz = meta["pending_quiz"]
                        break
        except Exception as e:
            logger.warning(f"Failed to restore pending_quiz from DB: {e}")

        # Load user profile
        try:
            user_profile = get_user_profile(user_id)
        except Exception as e:
            logger.warning(f"Failed to load user profile: {e}")
            user_profile = {
                "user_id": user_id,
                "learning_style": "conceptual",
                "preferred_tone": "encouraging",
                "academic_background": "대학원생",
                "notes": "",
            }

        state = DEFAULT_STATE.copy()
        state.update({
            "messages": initial_messages,
            "socratic_depth": request.socratic_depth,
            "session_id": session_id,
            "pending_quiz": restored_pending_quiz,
            "user_profile": user_profile,
            "selected_docs": request.selected_docs if request.selected_docs is not None else [],
        })
        
        runnable = GRAPH.compile()
        
        async def event_generator():
            current_state = state.copy()
            try:
                async for event in runnable.astream(current_state, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        current_state.update(node_output)
                        
                        # Serialize safely — 클라이언트에 노출할 필드만 선택
                        serializable_output = {}
                        for k, v in node_output.items():
                            if k in [
                                "rewritten_query", "route", "subtask", "active_agents",
                                "response", "evaluation", "retrieved_docs",
                                "tool_results", "frustration_level", "retry_count",
                                "tutor_response", "tool_result",
                            ]:
                                serializable_output[k] = v
                        
                        data = {
                            "type": "node_end",
                            "node": node_name,
                            "output": serializable_output
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
                # Send final result at the end (with tool_results)
                answer = current_state.get("response", "I'm sorry, I couldn't formulate a response.")
                
                # assistant 응답 DB 저장 (metadata 포함)
                metadata_dict = {}
                if current_state.get("tool_results"):
                    metadata_dict["tool_results"] = current_state.get("tool_results")
                if current_state.get("pending_quiz"):
                    metadata_dict["pending_quiz"] = current_state.get("pending_quiz")
                
                metadata_str = json.dumps(metadata_dict, ensure_ascii=False) if metadata_dict else None
                log_message(session_id, "assistant", answer, metadata=metadata_str)
                
                # 이번 턴에 generate_quiz가 실제로 호출된 경우에만 quiz_data 전송
                quiz_data = []
                for tr in current_state.get("tool_results", []):
                    if tr.get("tool") == "generate_quiz" and tr.get("ok"):
                        output = tr.get("output", {})
                        if isinstance(output, dict):
                            quiz_data = output.get("quiz", [])
                        break

                final_data = {
                    "type": "final_result",
                    "session_id": session_id,
                    "answer": answer,
                    "retrieved_docs": current_state.get("retrieved_docs", []),
                    "subtask": current_state.get("subtask"),
                    "route": current_state.get("route"),
                    "active_agents": current_state.get("active_agents", []),
                    "tool_results": current_state.get("tool_results", []),
                    "frustration_level": current_state.get("frustration_level", 0),
                    "socratic_depth": current_state.get("socratic_depth", 1),
                    "quiz_data": quiz_data,
                }
                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
                # 참고: 백그라운드 프로필 업데이트는 tool_agent 내부에서
                # threading.Thread(daemon=True)로 처리되므로 여기서 별도 트리거 불필요.
            except Exception as e:
                logger.error(f"[/chat] Stream Error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"
                
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"[/chat] ❌ Exception: {type(e).__name__}: {e}")
        logger.error("[/chat] Full traceback:\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, process it into chunks, and add to the vector store."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        # Save uploaded file temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"[/upload] Processing file: {file.filename}")
        
        # Process PDF into chunks
        processed_chunks = process_pdf(file_path)

        # Prepare for vector store
        file_hash = compute_file_hash(file_path)
        texts = [chunk["text"] for chunk in processed_chunks]
        metadatas = [chunk["metadata"] for chunk in processed_chunks]
        ids = [f"{file_hash[:16]}_{chunk['metadata']['chunk_index']}" for chunk in processed_chunks]

        # Add to vector store (중복 파일은 자동으로 스킵)
        num_added = add_documents(texts, metadatas=metadatas, ids=ids)
        
        if num_added == 0:
            logger.info(f"[/upload] Duplicate file skipped: {file.filename}")
            return {
                "filename": file.filename,
                "status": "duplicate",
                "chunks_added": 0
            }

        logger.info(f"[/upload] Successfully added {num_added} chunks from {file.filename}")
        return {
            "filename": file.filename,
            "status": "success",
            "chunks_added": num_added
        }
    except Exception as e:
        logger.error(f"[/upload] ❌ Error processing upload: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)


@app.get("/documents")
async def list_documents():
    """Get the list of all registered PDF filenames in the vector store."""
    try:
        docs = get_registered_documents()
        return {"documents": docs}
    except Exception as e:
        logger.error(f"[/documents] Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{filename}")
async def delete_pdf_document(filename: str):
    """Delete all chunks associated with a specific PDF filename from the vector store."""
    try:
        # Prevent path traversal just in case
        safe_filename = os.path.basename(filename)
        deleted_count = delete_document(safe_filename)
        return {
            "filename": safe_filename,
            "status": "success" if deleted_count > 0 else "not_found",
            "deleted_chunks": deleted_count
        }
    except Exception as e:
        logger.error(f"[/documents/{filename}] Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def get_sessions():
    """전체 세션 목록 반환 (최신순)."""
    try:
        sessions = list_sessions()
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"[/sessions] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    """세션 및 관련 메시지 전체 삭제."""
    try:
        delete_session(session_id)
        return {"status": "deleted", "session_id": session_id}
    except Exception as e:
        logger.error(f"[DELETE /sessions/{session_id}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """특정 세션의 대화 내역 반환."""
    try:
        msgs = get_messages(session_id)
        return {"session_id": session_id, "messages": msgs}
    except Exception as e:
        logger.error(f"[/sessions/{session_id}/messages] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weaknesses")
async def list_weaknesses():
    """미해결 약점 목록 반환 (최신순)."""
    try:
        items = get_weaknesses(resolved=False)
        return {"weaknesses": items}
    except Exception as e:
        logger.error(f"[/weaknesses] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/weaknesses/{weakness_id}")
async def remove_weakness(weakness_id: int):
    """약점 레코드 삭제."""
    try:
        deleted = delete_weakness(weakness_id)
        return {"status": "deleted" if deleted else "not_found", "weakness_id": weakness_id}
    except Exception as e:
        logger.error(f"[DELETE /weaknesses/{weakness_id}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schedules")
async def list_schedules():
    """미완료 복습 일정 목록 반환."""
    try:
        items = get_pending_schedules()
        return {"schedules": items}
    except Exception as e:
        logger.error(f"[/schedules] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/schedules/{schedule_id}")
async def remove_schedule(schedule_id: int):
    """복습 일정 삭제."""
    try:
        deleted = delete_schedule(schedule_id)
        return {"status": "deleted" if deleted else "not_found", "schedule_id": schedule_id}
    except Exception as e:
        logger.error(f"[DELETE /schedules/{schedule_id}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _parse_iso_dt(value: str) -> datetime:
    """Parse ISO datetime with light normalization for trailing Z."""
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _infer_category(concept: str) -> str:
    """Map concept text to a coarse study category for report stats."""
    text = (concept or "").lower()
    if any(k in text for k in ["트랜스포머", "어텐션", "인코더", "디코더", "포지셔널", "딥러닝", "신경망", "머신러닝"]):
        return "AI/딥러닝"
    if any(k in text for k in ["세마포어", "뮤텍스", "데드락", "스레드", "프로세스", "운영체제"]):
        return "운영체제"
    if any(k in text for k in ["cap", "분산", "일관성", "가용성", "partition", "consistency", "availability"]):
        return "분산시스템"
    if any(k in text for k in ["자료구조", "알고리즘", "정렬", "그래프", "트리", "dp"]):
        return "자료구조/알고리즘"
    return "기타"


@app.get("/recommend-chips")
async def recommend_chips():
    """Return personalized quick-start chips based on pending schedules and weaknesses."""
    try:
        now = datetime.now()
        horizon = now + timedelta(days=7)

        schedules = get_pending_schedules()
        weak_items = get_weaknesses(resolved=False)

        due_soon = []
        for s in schedules:
            review_at = s.get("review_at")
            if not review_at:
                continue
            try:
                dt = _parse_iso_dt(review_at)
            except Exception:
                continue
            if now <= dt <= horizon:
                due_soon.append((dt, s))

        due_soon.sort(key=lambda item: item[0])
        weak_sorted = sorted(weak_items, key=lambda w: int(w.get("severity") or 1), reverse=True)

        chips: List[Dict[str, Any]] = []

        for dt, s in due_soon[:2]:
            desc = (s.get("description") or "복습 일정").strip()
            label_prefix = "🚨 오늘" if dt.date() == now.date() else "🔁 복습"
            chips.append(
                {
                    "label": f"{label_prefix}: {desc[:20]}",
                    "message": f"{desc}를 오늘 복습할 수 있게 소크라테스식으로 도와줘",
                    "type": "schedule",
                    "due": dt.date().isoformat(),
                }
            )

        for w in weak_sorted[:2]:
            concept = (w.get("concept") or "핵심 개념").strip()
            chips.append(
                {
                    "label": f"⚠️ 약점 보완: {concept[:20]}",
                    "message": f"{concept} 개념을 내가 이해했는지 점검하면서 복습해줘",
                    "type": "weakness",
                }
            )

        # No fallback chips by design: if there is no personalized data, return empty list.
        return {"chips": chips[:4]}
    except Exception as e:
        logger.error(f"[/recommend-chips] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report-data")
async def get_report_data(user_id: str = "default"):
    """Aggregate strengths, weaknesses, profile, and stats for metacognition report UI."""
    try:
        weaknesses = get_weaknesses(resolved=False, limit=100)
        strengths = get_user_strengths(user_id=user_id, limit=100)
        profile = get_user_profile(user_id=user_id)

        now = datetime.now()
        recent_7d_weaknesses = 0
        category_counter: Dict[str, int] = {}
        for w in weaknesses:
            created_at = w.get("created_at")
            if created_at:
                try:
                    if _parse_iso_dt(created_at) >= (now - timedelta(days=7)):
                        recent_7d_weaknesses += 1
                except Exception:
                    pass
            category = _infer_category(w.get("concept", ""))
            category_counter[category] = category_counter.get(category, 0) + 1

        top_categories = [k for k, _ in sorted(category_counter.items(), key=lambda kv: kv[1], reverse=True)[:3]]

        stats = {
            "total_weaknesses": len(weaknesses),
            "total_strengths": len(strengths),
            "recent_7d_weaknesses": recent_7d_weaknesses,
            "top_categories": top_categories,
        }

        return {
            "weaknesses": weaknesses,
            "strengths": strengths,
            "profile_summary": {
                "strengths_summary": profile.get("strengths_summary", ""),
                "weaknesses_summary": profile.get("weaknesses_summary", ""),
                "learning_style": profile.get("learning_style", "conceptual"),
            },
            "stats": stats,
        }
    except Exception as e:
        logger.error(f"[/report-data] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports")
async def list_reports(user_id: str = "default", limit: int = 10):
    """Return saved metacognition reports."""
    try:
        return {"reports": get_reports(user_id=user_id, limit=limit)}
    except Exception as e:
        logger.error(f"[/reports] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-report")
async def generate_report(payload: dict | None = None):
    """Generate a markdown report from learner profile/strength/weakness context and save it."""
    payload = payload or {}
    user_id = payload.get("user_id", "default")

    try:
        report_data = await get_report_data(user_id=user_id)

        weaknesses = report_data.get("weaknesses", [])
        strengths = report_data.get("strengths", [])
        profile_summary = report_data.get("profile_summary", {})
        stats = report_data.get("stats", {})

        strengths_lines = [f"- {s.get('concept', '개념')}" for s in strengths[:12]] or ["- (기록 없음)"]
        weakness_lines = [
            f"- {w.get('concept', '개념')} (심각도 {w.get('severity', 1)})"
            for w in weaknesses[:12]
        ] or ["- (기록 없음)"]

        prompt = (
            "당신은 학습 코치입니다. 아래 학습 데이터를 바탕으로 한국어 메타인지 리포트를 마크다운으로 작성하세요.\n"
            "길이는 500~900자 내외로 간결하지만 실행 가능해야 합니다.\n\n"
            "[학습 데이터]\n"
            f"강점 목록:\n{chr(10).join(strengths_lines)}\n\n"
            f"약점 목록:\n{chr(10).join(weakness_lines)}\n\n"
            f"강점 요약: {profile_summary.get('strengths_summary', '')}\n"
            f"약점 요약: {profile_summary.get('weaknesses_summary', '')}\n"
            f"학습 스타일: {profile_summary.get('learning_style', 'conceptual')}\n"
            f"통계: total_weaknesses={stats.get('total_weaknesses', 0)}, total_strengths={stats.get('total_strengths', 0)}, recent_7d_weaknesses={stats.get('recent_7d_weaknesses', 0)}, top_categories={stats.get('top_categories', [])}\n\n"
            "[출력 형식]\n"
            "## ✅ 잘 이해하고 있는 것\n"
            "## 🔴 보완이 필요한 것\n"
            "## 💡 다음 학습 권장 순서 (3단계)\n"
            "## 📈 총평\n"
        )

        generated_body = ""
        try:
            llm_response = llm.invoke(prompt)
            generated_body = getattr(llm_response, "content", str(llm_response)).strip()
        except Exception as llm_err:
            logger.warning(f"[/generate-report] LLM generation failed, using fallback: {llm_err}")

        if not generated_body:
            generated_body = (
                "## ✅ 잘 이해하고 있는 것\n"
                + ("\n".join(strengths_lines[:5]) if strengths else "- 아직 명시적인 강점 기록이 없습니다.")
                + "\n\n## 🔴 보완이 필요한 것\n"
                + ("\n".join(weakness_lines[:5]) if weaknesses else "- 현재 등록된 약점이 없습니다.")
                + "\n\n## 💡 다음 학습 권장 순서 (3단계)\n"
                + "1. 심각도 높은 약점 1개를 선택해 개념 정의를 스스로 설명해보기\n"
                + "2. 선택한 약점으로 3문항 퀴즈를 풀고 오답 이유를 정리하기\n"
                + "3. 48시간 내 같은 개념을 다시 복습해 장기기억으로 전환하기\n"
                + "\n## 📈 총평\n"
                + f"약점 {stats.get('total_weaknesses', 0)}개, 강점 {stats.get('total_strengths', 0)}개 기반으로 다음 학습 우선순위를 재정렬해야 합니다."
            )

        title = f"메타인지 리포트 ({datetime.now().strftime('%Y-%m-%d')})"
        report_id = save_report(title=title, body=generated_body, user_id=user_id)

        return {
            "report_id": report_id,
            "title": title,
            "body": generated_body,
        }
    except Exception as e:
        logger.error(f"[/generate-report] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit_quiz")
async def submit_quiz(payload: dict):
    """퀴즈 답안 제출 및 자동 채점.
    
    요청:
    {
        "quiz_items": [
            {"question": "...", "options": [...], "answer": "A"},
            ...
        ],
        "user_answers": {
            "0": "A",
            "1": "B",
            ...
        }
    }
    
    응답:
    {
        "score": 80,
        "correct": 4,
        "total": 5,
        "message": "✅ 4/5 정답입니다! 점수: 80점\n✅ 1번: 정답\n✅ 2번: 정답\n...",
        "details": ["✅ 1번: 정답", ...],
        "suggest_weakness": False
    }
    """
    try:
        from src.tools.learning_tools import grade_quiz, save_strength

        quiz_items = payload.get("quiz_items", [])
        user_answers_raw = payload.get("user_answers", {})
        topic = payload.get("topic", "")
        session_id = payload.get("session_id")
        user_id = payload.get("user_id", "default")

        # Convert user_answers keys to int ("0" -> 0)
        user_answers = {}
        for key, val in user_answers_raw.items():
            try:
                user_answers[int(key)] = val.upper()
            except (ValueError, AttributeError):
                pass

        logger.info(f"[/submit_quiz] Grading {len(quiz_items)} questions, {len(user_answers)} answers")

        result = grade_quiz(quiz_items, user_answers)

        logger.info(f"[/submit_quiz] Result: score={result.get('score')}, suggest_weakness={result.get('suggest_weakness')}")

        if result.get("score", 0) >= 80 and topic:
            strength_id = save_strength({
                "concept": topic,
                "details": f"퀴즈 {result['score']}점 달성 ({result['correct']}/{result['total']} 정답)",
                "session_id": session_id,
                "user_id": user_id,
            })
            result["strength_saved"] = True
            result["strength_id"] = strength_id
            logger.info(f"[/submit_quiz] save_strength called: topic={topic}, score={result['score']}, id={strength_id}")
        else:
            result["strength_saved"] = False

        return result

    except Exception as e:
        logger.error(f"[/submit_quiz] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
