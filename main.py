"""
SHL Assessment Recommender -- FastAPI service.
Endpoints (exact spec from assignment):
  GET  /health  -> {"status": "ok"}
  POST /chat    -> {"reply": str, "recommendations": [...], "end_of_conversation": bool}
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent import run_agent_turn

app = FastAPI(title="SHL Assessment Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    result = run_agent_turn(messages)
    return JSONResponse(content=result)
