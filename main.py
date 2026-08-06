"""
SHL Assessment Recommender -- FastAPI service.
Endpoints:
  GET  /        -> redirects to /docs
  GET  /health  -> {"status": "ok"}
  POST /chat    -> {"reply": str, "recommendations": [...], "end_of_conversation": bool}
"""

from dotenv import load_dotenv
load_dotenv()

from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from agent import run_agent_turn

app = FastAPI(title="SHL Assessment Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": "user", "content": "Hiring a mid-level Java developer"}
                ]
            }
        }
    }


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(payload: ChatRequest):
    messages = [m.model_dump() for m in payload.messages]
    result = run_agent_turn(messages)
    return JSONResponse(content=result)