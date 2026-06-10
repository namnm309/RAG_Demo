from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from openai import OpenAI

from config import CONFIG
from models.schemas import ChatRequest, GenerateQuestionsRequest
from services.ingestion import DocumentIngestionService
from services.interview_generator import InterviewQuestionService
from services.rag_chat import RagChatService
from services.vector_store import ChromaVectorStore

client = OpenAI(api_key="ollama", base_url=CONFIG["ollama_base_url"])
vector_store = ChromaVectorStore(
    persist_dir=CONFIG["chroma_persist_dir"],
    system_collection=CONFIG["chroma_system_collection"],
    hr_collection=CONFIG["chroma_hr_collection"],
)
ingestion_service = DocumentIngestionService(vector_store, client, CONFIG)
rag_service = RagChatService(vector_store, client, CONFIG)
interview_service = InterviewQuestionService(vector_store, client, CONFIG)

app = FastAPI(title="IQGS RAG Service")


def verify_internal_api_key(
    x_internal_api_key: Optional[str] = Header(default=None),
):
    configured_key = CONFIG.get("internal_api_key", "")
    if not configured_key:
        return
    if x_internal_api_key != configured_key:
        raise HTTPException(status_code=401, detail="Invalid internal API key")


@app.get("/health")
def health():
    system_chunks = vector_store.count("system")
    hr_chunks = vector_store.count("hr")
    base = {
        "chat_model": CONFIG["chat_model"],
        "embedding_model": CONFIG["embedding_model"],
        "ollama_base_url": CONFIG["ollama_base_url"],
        "system_chunks": system_chunks,
        "hr_chunks": hr_chunks,
    }
    try:
        client.models.list()
        return {"status": "ok", **base}
    except Exception as exc:
        return {
            "status": "error",
            "message": f"Cannot connect to Ollama: {exc}",
            **base,
        }


@app.get("/status", dependencies=[Depends(verify_internal_api_key)])
def status():
    return {
        "system_chunks": vector_store.count("system"),
        "hr_chunks": vector_store.count("hr"),
        "system_files": vector_store.indexed_files("system"),
        "hr_files": vector_store.indexed_files("hr"),
    }


@app.post("/chat", dependencies=[Depends(verify_internal_api_key)])
def chat(request: ChatRequest):
    response = rag_service.ask(request)
    return response.to_json_dict()


@app.post("/generate-questions", dependencies=[Depends(verify_internal_api_key)])
def generate_questions(request: GenerateQuestionsRequest):
    response = interview_service.generate(request)
    return response.to_json_dict()


@app.post("/ingest/system", dependencies=[Depends(verify_internal_api_key)])
def ingest_system():
    return ingestion_service.ingest_system()


@app.post("/ingest/hr/{owner_id}", dependencies=[Depends(verify_internal_api_key)])
def ingest_hr(owner_id: str):
    return ingestion_service.ingest_hr(owner_id)
