from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.converters import to_generate_questions_request
from api.deps import get_services, verify_api_key
from api.mappers import map_http_status, to_questions_response
from models.api.v1.common import ApiResponse, ErrorBody, ErrorCode
from models.api.v1.questions import GenerateQuestionsRequestBody, QuestionsData

router = APIRouter(tags=["Interview Questions"])


@router.post(
    "/interview-questions",
    response_model=ApiResponse[QuestionsData],
    summary="Generate interview questions",
    dependencies=[Depends(verify_api_key)],
)
def generate_questions(body: GenerateQuestionsRequestBody):
    services = get_services()
    request = to_generate_questions_request(body)
    if request.confirmed_plan:
        domain = services.interview_service.generate_from_plan(request.confirmed_plan)
    elif request.owner_id and request.role and request.level:
        domain = services.interview_service.generate(request)
    else:
        envelope = ApiResponse(
            success=False,
            error=ErrorBody(
                code=ErrorCode.VALIDATION_ERROR,
                message="Cần confirmed_plan hoặc owner_id/role/level.",
            ),
        )
        return JSONResponse(status_code=400, content=envelope.model_dump())

    envelope = to_questions_response(domain)
    code = envelope.error.code if envelope.error else None
    return JSONResponse(
        status_code=map_http_status(envelope.success, code),
        content=envelope.model_dump(),
    )
