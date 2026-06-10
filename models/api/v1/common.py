from enum import Enum
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    JD_INVALID = "JD_INVALID"
    INGEST_FAILED = "INGEST_FAILED"
    PLAN_ERROR = "PLAN_ERROR"
    LLM_ERROR = "LLM_ERROR"
    NOT_READY = "NOT_READY"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorBody(BaseModel):
    code: ErrorCode
    message: str
    details: List[str] = Field(default_factory=list)


class Meta(BaseModel):
    processing_time_ms: Optional[float] = None
    session_id: Optional[str] = None
    phase: Optional[str] = None
    conversation_id: Optional[str] = None


T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: Optional[T] = None
    error: Optional[ErrorBody] = None
    meta: Meta = Field(default_factory=Meta)


class SourceRef(BaseModel):
    file_name: str
    chunk_index: int
    score: float
    text_preview: str
    knowledge_base: str = "system"
    doc_type: str = "general"


class ChatMessageBody(BaseModel):
    role: str
    content: str
    created_at: Optional[str] = None


class InterviewPlanBody(BaseModel):
    owner_id: str
    role: str
    level: str
    question_count: int = 5
    question_types: List[str] = Field(default_factory=lambda: ["technical", "behavioral"])
    topics: List[str] = Field(default_factory=list)
    summary: str = ""
    constraints: str = ""
    notes: str = ""
