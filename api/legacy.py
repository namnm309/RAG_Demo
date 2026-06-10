from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.deps import get_services, verify_api_key
from models.schemas import ChatRequest, GeneratePlanRequest, GenerateQuestionsRequest

router = APIRouter(tags=["Legacy (deprecated)"])

DEPRECATION_HEADERS = {
    "Deprecation": "true",
    "Sunset": "2026-09-01",
    "Link": '</api/v1/docs>; rel="successor-version"',
}


def _deprecated(content: dict):
    from fastapi.responses import JSONResponse

    return JSONResponse(content=content, headers=DEPRECATION_HEADERS)


async def _read_uploads(files: List[UploadFile]) -> list[tuple[str, bytes]]:
    uploads: list[tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        uploads.append((upload.filename or "", content))
    return uploads


@router.get("/status", dependencies=[Depends(verify_api_key)], include_in_schema=False)
def legacy_status():
    services = get_services()
    return _deprecated(
        {
            "system_chunks": services.vector_store.count("system"),
            "hr_chunks": services.vector_store.count("hr"),
            "system_files": services.vector_store.indexed_files("system"),
            "hr_files": services.vector_store.indexed_files("hr"),
        }
    )


@router.post("/chat", dependencies=[Depends(verify_api_key)], include_in_schema=False)
def legacy_chat(request: ChatRequest):
    return _deprecated(get_services().rag_service.ask(request).to_json_dict())


@router.post("/generate-plan", dependencies=[Depends(verify_api_key)], include_in_schema=False)
def legacy_generate_plan(request: GeneratePlanRequest):
    return _deprecated(get_services().plan_service.handle(request).to_json_dict())


@router.post("/generate-questions", dependencies=[Depends(verify_api_key)], include_in_schema=False)
def legacy_generate_questions(request: GenerateQuestionsRequest):
    services = get_services()
    if request.confirmed_plan:
        response = services.interview_service.generate_from_plan(request.confirmed_plan)
    else:
        response = services.interview_service.generate(request)
    return _deprecated(response.to_json_dict())


@router.post("/ingest/system", dependencies=[Depends(verify_api_key)], include_in_schema=False)
async def legacy_ingest_system(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Cần ít nhất một file trong field 'files'")
    uploads = await _read_uploads(files)
    return _deprecated(
        get_services().ingestion_service.ingest_uploaded_files("system", uploads).to_json_dict()
    )


@router.post("/ingest/hr/{owner_id}", dependencies=[Depends(verify_api_key)], include_in_schema=False)
async def legacy_ingest_hr(owner_id: str, files: List[UploadFile] = File(...)):
    if not owner_id or not owner_id.strip():
        raise HTTPException(status_code=400, detail="owner_id không hợp lệ")
    if not files:
        raise HTTPException(status_code=400, detail="Cần ít nhất một file trong field 'files'")
    uploads = await _read_uploads(files)
    return _deprecated(
        get_services()
        .ingestion_service.ingest_uploaded_files("hr", uploads, owner_id=owner_id.strip())
        .to_json_dict()
    )
