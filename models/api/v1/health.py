from typing import List, Literal, Optional

from pydantic import BaseModel


class HealthData(BaseModel):
    status: Literal["ok", "error"]
    chat_model: str
    embedding_model: str
    ollama_base_url: str
    system_chunks: int
    hr_chunks: int
    message: Optional[str] = None


class StatusData(BaseModel):
    system_chunks: int
    hr_chunks: int
    system_files: List[str]
    hr_files: List[str]
