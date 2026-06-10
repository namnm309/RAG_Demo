from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional


KnowledgeBase = Literal["system", "hr"]
CollectionName = Literal["system", "hr"]


@dataclass
class DocumentChunk:
    id: str
    text: str
    source_file: str
    chunk_index: int
    embedding: List[float]
    knowledge_base: KnowledgeBase = "system"
    owner_id: Optional[str] = None
    doc_type: str = "general"
    score: float = 0.0


@dataclass
class IngestResponse:
    success: bool
    message: str
    documents_loaded: int = 0
    chunks_created: int = 0
    files: List[str] = field(default_factory=list)


@dataclass
class ChatMessage:
    role: str
    content: str
    created_at: Optional[str] = None


@dataclass
class ChatRequest:
    question: str
    top_k: int = 5
    owner_id: Optional[str] = None
    conversation_id: Optional[str] = None
    chat_history: List[ChatMessage] = field(default_factory=list)


@dataclass
class SourceReference:
    file_name: str
    chunk_index: int
    score: float
    text_preview: str
    knowledge_base: str = "system"
    doc_type: str = "general"


@dataclass
class ChatResponse:
    answer: str
    sources: List[SourceReference] = field(default_factory=list)
    chunks_used: int = 0
    processing_time_ms: float = 0.0


@dataclass
class GenerateQuestionsRequest:
    owner_id: str
    role: str
    level: str
    question_count: int = 5
    question_types: List[str] = field(default_factory=lambda: ["technical", "behavioral"])
    extra_context: str = ""
    top_k_system: int = 0
    top_k_hr: int = 0


@dataclass
class QuestionCitation:
    knowledge_base: str
    source_file: str
    chunk_index: int
    excerpt: str


@dataclass
class GeneratedQuestion:
    question: str
    question_type: str
    difficulty: str
    rationale: str
    sample_answer: str = ""
    citations: List[QuestionCitation] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


@dataclass
class GenerateQuestionsResponse:
    success: bool
    questions: List[GeneratedQuestion] = field(default_factory=list)
    raw_answer: str = ""
    sources: List[SourceReference] = field(default_factory=list)
    processing_time_ms: float = 0.0
    error: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "error": self.error,
            "processing_time_ms": self.processing_time_ms,
            "questions": [
                {
                    "question": q.question,
                    "question_type": q.question_type,
                    "difficulty": q.difficulty,
                    "rationale": q.rationale,
                    "sample_answer": q.sample_answer,
                    "citations": [asdict(c) for c in q.citations],
                    "sources": q.sources,
                }
                for q in self.questions
            ],
            "retrieval_sources": [
                {
                    "file_name": s.file_name,
                    "chunk_index": s.chunk_index,
                    "score": s.score,
                    "knowledge_base": s.knowledge_base,
                    "doc_type": s.doc_type,
                    "text_preview": s.text_preview,
                }
                for s in self.sources
            ],
        }
