from typing import List, Optional

from pydantic import BaseModel, Field

from models.api.v1.common import InterviewPlanBody, SourceRef


class QuestionCitationData(BaseModel):
    knowledge_base: str
    source_file: str
    chunk_index: int
    excerpt: str


class GeneratedQuestionData(BaseModel):
    question: str
    question_type: str
    difficulty: str
    rationale: str
    sample_answer: str = ""
    citations: List[QuestionCitationData] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class GenerateQuestionsRequestBody(BaseModel):
    owner_id: str = ""
    role: str = ""
    level: str = ""
    question_count: int = 5
    question_types: List[str] = Field(default_factory=lambda: ["technical", "behavioral"])
    extra_context: str = ""
    top_k_system: int = 0
    top_k_hr: int = 0
    confirmed_plan: Optional[InterviewPlanBody] = None


class QuestionsData(BaseModel):
    questions: List[GeneratedQuestionData] = Field(default_factory=list)
    sources: List[SourceRef] = Field(default_factory=list)
    raw_answer: str = ""
