"""
Pydantic models matching the assignment's API spec EXACTLY.
The schema is non-negotiable per the assignment doc -- deviations break
the automated evaluator.
"""

from pydantic import BaseModel
from typing import Literal, Optional


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


class HealthResponse(BaseModel):
    status: str
