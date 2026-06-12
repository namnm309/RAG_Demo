from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.deps import get_services, verify_api_key
from api.mappers import map_http_status, to_health_response, to_status_response
from config import CONFIG
from models.api.v1.common import ApiResponse
from models.api.v1.health import HealthData, StatusData

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=ApiResponse[HealthData],
    summary="Check thông tin rag ",
    description="Kiểm tra Ollama và số chunk trong ChromaDB",
)
def health():
    services = get_services()
    system_chunks = services.vector_store.count("system")
    hr_chunks = services.vector_store.count("hr")
    base_kwargs = dict(
        chat_model=CONFIG["chat_model"],
        embedding_model=CONFIG["embedding_model"],
        ollama_base_url=CONFIG["ollama_base_url"],
        system_chunks=system_chunks,
        hr_chunks=hr_chunks,
    )
    try:
        services.client.models.list()
        envelope = to_health_response(status="ok", **base_kwargs)
    except Exception as exc:
        envelope = to_health_response(
            status="error",
            message=f"Cannot connect to Ollama: {exc}",
            **base_kwargs,
        )
    status_code = map_http_status(
        envelope.success,
        envelope.error.code if envelope.error else None,
    )
    return JSONResponse(status_code=status_code, content=envelope.model_dump())


@router.get(
    "/status",
    response_model=ApiResponse[StatusData],
    summary="Xem các file đã nạp ",
    dependencies=[Depends(verify_api_key)],
)
def status():
    services = get_services()
    envelope = to_status_response(
        system_chunks=services.vector_store.count("system"),
        hr_chunks=services.vector_store.count("hr"),
        system_files=services.vector_store.indexed_files("system"),
        hr_files=services.vector_store.indexed_files("hr"),
    )
    return JSONResponse(status_code=200, content=envelope.model_dump())
