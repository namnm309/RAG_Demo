from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.deps import get_services, verify_api_key
from api.mappers import map_http_status, to_ingest_response
from models.api.v1.common import ApiResponse, ErrorBody, ErrorCode
from models.api.v1.ingest import IngestData

router = APIRouter(prefix="/knowledge", tags=["Knowledge"])


async def _read_uploads(files: List[UploadFile]) -> list[tuple[str, bytes]]:
    uploads: list[tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        uploads.append((upload.filename or "", content))
    return uploads


def _ingest_json_response(envelope: ApiResponse[IngestData]) -> JSONResponse:
    code = envelope.error.code if envelope.error else None
    return JSONResponse(
        status_code=map_http_status(envelope.success, code),
        content=envelope.model_dump(),
    )


@router.post(
    "/system/files",
    response_model=ApiResponse[IngestData],
    summary="nạp file tài liệu",
    dependencies=[Depends(verify_api_key)],
)
async def ingest_system(files: List[UploadFile] = File(...)):
    if not files:
        envelope = ApiResponse(
            success=False,
            error=ErrorBody(
                code=ErrorCode.VALIDATION_ERROR,
                message="Cần ít nhất một file trong field 'files'",
            ),
        )
        return JSONResponse(status_code=400, content=envelope.model_dump())
    uploads = await _read_uploads(files)
    result = get_services().ingestion_service.ingest_uploaded_files("system", uploads)
    return _ingest_json_response(to_ingest_response(result))


@router.post(
    "/hr/{owner_id}/files",
    response_model=ApiResponse[IngestData],
    summary="nạp file HR JD",
    dependencies=[Depends(verify_api_key)],
)
async def ingest_hr(owner_id: str, files: List[UploadFile] = File(...)):
    if not owner_id or not owner_id.strip():
        raise HTTPException(status_code=400, detail="owner_id không hợp lệ")
    if not files:
        envelope = ApiResponse(
            success=False,
            error=ErrorBody(
                code=ErrorCode.VALIDATION_ERROR,
                message="Cần ít nhất một file trong field 'files'",
            ),
        )
        return JSONResponse(status_code=400, content=envelope.model_dump())
    uploads = await _read_uploads(files)
    result = get_services().ingestion_service.ingest_uploaded_files(
        "hr", uploads, owner_id=owner_id.strip()
    )
    return _ingest_json_response(to_ingest_response(result))
