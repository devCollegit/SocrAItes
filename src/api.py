from fastapi import FastAPI, HTTPException, Request, UploadFile, File
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

app = FastAPI(title="SocrAItes API")

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

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"[/chat] session={session_id} | messages={len(request.messages)} | depth={request.socratic_depth}")
    try:
        initial_messages = [{"role": m.role, "content": m.content} for m in request.messages]
        logger.debug(f"[/chat] last user message: {initial_messages[-1]['content'][:100] if initial_messages else '(empty)'}")
        
        state = DEFAULT_STATE.copy()
        state.update({
            "messages": initial_messages,
            "socratic_depth": request.socratic_depth,
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
                            if k in ["contextualized_query", "next_step", "plan", "draft_answer", "evaluation", "retrieved_docs"]:
                                serializable_output[k] = v
                        
                        data = {
                            "type": "node_end",
                            "node": node_name,
                            "output": serializable_output
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
                # Send final result at the end
                final_data = {
                    "type": "final_result",
                    "session_id": session_id,
                    "answer": current_state.get("draft_answer", "I'm sorry, I couldn't formulate a response."),
                    "retrieved_docs": current_state.get("retrieved_docs", []),
                    "plan": current_state.get("plan")
                }
                yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
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


@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
