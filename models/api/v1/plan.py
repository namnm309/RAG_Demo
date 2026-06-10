from typing import List, Optional

from pydantic import BaseModel, Field

from models.api.v1.common import ChatMessageBody, InterviewPlanBody, SourceRef


class PlanStartRequestBody(BaseModel):
    owner_id: str
    session_id: Optional[str] = None


class PlanMessageRequestBody(BaseModel):
    owner_id: str
    message: str
    session_id: Optional[str] = None
    chat_history: List[ChatMessageBody] = Field(default_factory=list)


class PlanConfirmRequestBody(BaseModel):
    owner_id: str
    plan_draft: InterviewPlanBody
    session_id: Optional[str] = None
    chat_history: List[ChatMessageBody] = Field(default_factory=list)


class PlanData(BaseModel):
    assistant_message: str = ""
    clarifying_questions: List[str] = Field(default_factory=list)
    plan: Optional[InterviewPlanBody] = None
    validation_errors: List[str] = Field(default_factory=list)
    sources: List[SourceRef] = Field(default_factory=list)
    raw_answer: str = ""
