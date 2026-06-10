from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException
from openai import OpenAI

from config import CONFIG
from services.ingestion import DocumentIngestionService
from services.interview_generator import InterviewQuestionService
from services.interview_plan import InterviewPlanService
from services.rag_chat import RagChatService
from services.vector_store import ChromaVectorStore


@dataclass
class AppServices:
    client: OpenAI
    vector_store: ChromaVectorStore
    ingestion_service: DocumentIngestionService
    rag_service: RagChatService
    interview_service: InterviewQuestionService
    plan_service: InterviewPlanService


_client = OpenAI(api_key="ollama", base_url=CONFIG["ollama_base_url"])
_vector_store = ChromaVectorStore(
    persist_dir=CONFIG["chroma_persist_dir"],
    system_collection=CONFIG["chroma_system_collection"],
    hr_collection=CONFIG["chroma_hr_collection"],
)
_ingestion_service = DocumentIngestionService(_vector_store, _client, CONFIG)
_rag_service = RagChatService(_vector_store, _client, CONFIG)
_interview_service = InterviewQuestionService(_vector_store, _client, CONFIG)
_plan_service = InterviewPlanService(_vector_store, _ingestion_service, _client, CONFIG)

SERVICES = AppServices(
    client=_client,
    vector_store=_vector_store,
    ingestion_service=_ingestion_service,
    rag_service=_rag_service,
    interview_service=_interview_service,
    plan_service=_plan_service,
)


def get_services() -> AppServices:
    return SERVICES


def verify_api_key(
    x_internal_api_key: Optional[str] = Header(default=None, alias="X-Internal-Api-Key"),
) -> None:
    configured_key = CONFIG.get("internal_api_key", "")
    if not configured_key:
        return
    if x_internal_api_key != configured_key:
        raise HTTPException(status_code=401, detail="Invalid internal API key")
