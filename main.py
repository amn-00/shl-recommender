"""
SHL Assessment Recommender -- FastAPI service.
Endpoints (exact spec from assignment):
  GET  /health  -> {"status": "ok"}
  POST /chat    -> {"reply": str, "recommendations": [...], "end_of_conversation": bool}
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schemas import ChatRequest, ChatResponse, HealthResponse
from agent import run_agent_turn

app = FastAPI(title="SHL Assessment Recommender")

# Permissive CORS since the evaluator harness calls this from an external host.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    result = run_agent_turn(messages)
    return ChatResponse(**result)
