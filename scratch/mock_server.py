
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import uvicorn

app = FastAPI()

# Mount static files from src/static
static_path = os.path.abspath("src/static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_path, "index.html"))

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_msg = data["messages"][-1]["content"]
    
    # Mock response
    return {
        "answer": f"소크라테스식 답변 (Mock): '{user_msg}'에 대해 스스로 어떻게 생각하시나요? 어떤 근거로 그렇게 판단하셨는지 말씀해 주실 수 있을까요?",
        "session_id": "mock-session",
        "plan": "사용자의 질문을 분석하고 개념적 전제를 검토하는 중입니다.",
        "retrieved_docs": []
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
