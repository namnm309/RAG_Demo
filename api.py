from typing import List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from openai import OpenAI

from config import CONFIG
from models.schemas import ChatRequest, GeneratePlanRequest, GenerateQuestionsRequest
from services.ingestion import DocumentIngestionService
from services.interview_generator import InterviewQuestionService
from services.interview_plan import InterviewPlanService
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
plan_service = InterviewPlanService(vector_store, ingestion_service, client, CONFIG)

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


@app.post("/generate-plan", dependencies=[Depends(verify_internal_api_key)])
def generate_plan(request: GeneratePlanRequest):
    response = plan_service.handle(request)
    return response.to_json_dict()


@app.post("/generate-questions", dependencies=[Depends(verify_internal_api_key)])
def generate_questions(request: GenerateQuestionsRequest):
    if request.confirmed_plan:
        response = interview_service.generate_from_plan(request.confirmed_plan)
    else:
        response = interview_service.generate(request)
    return response.to_json_dict()


async def _read_uploads(files: List[UploadFile]) -> list[tuple[str, bytes]]:
    uploads: list[tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        uploads.append((upload.filename or "", content))
    return uploads


@app.post("/ingest/system", dependencies=[Depends(verify_internal_api_key)])
async def ingest_system(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Cần ít nhất một file trong field 'files'")
    uploads = await _read_uploads(files)
    return ingestion_service.ingest_uploaded_files("system", uploads).to_json_dict()


@app.post("/ingest/hr/{owner_id}", dependencies=[Depends(verify_internal_api_key)])
async def ingest_hr(owner_id: str, files: List[UploadFile] = File(...)):
    if not owner_id or not owner_id.strip():
        raise HTTPException(status_code=400, detail="owner_id không hợp lệ")
    if not files:
        raise HTTPException(status_code=400, detail="Cần ít nhất một file trong field 'files'")
    uploads = await _read_uploads(files)
    return ingestion_service.ingest_uploaded_files(
        "hr", uploads, owner_id=owner_id.strip()
    ).to_json_dict()
