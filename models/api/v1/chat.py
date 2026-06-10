from typing import List, Optional

from pydantic import BaseModel, Field

from models.api.v1.common import ChatMessageBody, SourceRef


class ChatRequestBody(BaseModel):
    question: str
    top_k: int = 5
    owner_id: Optional[str] = None
    conversation_id: Optional[str] = None
    chat_history: List[ChatMessageBody] = Field(default_factory=list)


class ChatData(BaseModel):
    answer: str
    sources: List[SourceRef] = Field(default_factory=list)
    chunks_used: int = 0
