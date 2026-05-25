from fastapi import FastAPI, HTTPException, Request, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
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
from .rag.document_processor import process_pdf, compute_file_hash
from .rag.vectorstore import add_documents, get_registered_documents, delete_document
from .db.database import init_db, create_session, log_message, get_messages, list_sessions, delete_session, get_weaknesses, delete_weakness, get_pending_schedules, delete_schedule, get_user_profile, get_session

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

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    retrieved_docs: List[Any] = []
    plan: Optional[str] = None
    tool_results: List[Any] = []

@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
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
        })
        
        runnable = GRAPH.compile()
        
        async def event_generator():
            current_state = state.copy()
            try:
                async for event in runnable.astream(current_state, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        current_state.update(node_output)
                        
                        # Serialize safely
                        serializable_output = {}
                        for k, v in node_output.items():
                            if k in ["contextualized_query", "next_step", "plan", "sub_agents", "draft_answer", "evaluation", "retrieved_docs", "tool_results", "frustration_level", "retry_count"]:
                                serializable_output[k] = v
                        
                        data = {
                            "type": "node_end",
                            "node": node_name,
                            "output": serializable_output
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
                # Send final result at the end (with tool_results)
                answer = current_state.get("draft_answer", "I'm sorry, I couldn't formulate a response.")
                
                # assistant 응답 DB 저장 (metadata 포함)
                metadata_dict = {}
                if current_state.get("tool_results"):
                    metadata_dict["tool_results"] = current_state.get("tool_results")
                if current_state.get("pending_quiz"):
                    metadata_dict["pending_quiz"] = current_state.get("pending_quiz")
                
                metadata_str = json.dumps(metadata_dict, ensure_ascii=False) if metadata_dict else None
                log_message(session_id, "assistant", answer, metadata=metadata_str)
                
                final_data = {
                    "type": "final_result",
                    "session_id": session_id,
                    "answer": answer,
                    "retrieved_docs": current_state.get("retrieved_docs", []),
                    "plan": current_state.get("plan"),
                    "tool_results": current_state.get("tool_results", []),
                    "frustration_level": current_state.get("frustration_level", 0)
                }
                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"

                # Trigger background diagnosis asynchronously after SSE ends
                try:
                    full_history = []
                    for m in initial_messages:
                        full_history.append({"role": m["role"], "content": m["content"]})
                    full_history.append({"role": "assistant", "content": answer})
                    
                    from src.agent.sub_agents.diagnosis import run_background_diagnosis
                    background_tasks.add_task(run_background_diagnosis, session_id, user_id, full_history)
                    logger.info(f"[/chat] Enqueued background diagnosis task for session={session_id}, user={user_id}")
                except Exception as e:
                    logger.warning(f"Failed to queue background diagnosis: {e}")
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
        from src.tools.learning_tools import grade_quiz
        
        quiz_items = payload.get("quiz_items", [])
        user_answers_raw = payload.get("user_answers", {})
        
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
