from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.converters import (
    to_plan_confirm_request,
    to_plan_message_request,
    to_plan_start_request,
)
from api.deps import get_services, verify_api_key
from api.mappers import map_http_status, to_plan_response
from models.api.v1.common import ApiResponse
from models.api.v1.plan import (
    PlanConfirmRequestBody,
    PlanData,
    PlanMessageRequestBody,
    PlanStartRequestBody,
)

router = APIRouter(prefix="/interview-plans", tags=["Interview Plans"])


def _plan_json_response(envelope: ApiResponse[PlanData]) -> JSONResponse:
    code = envelope.error.code if envelope.error else None
    return JSONResponse(
        status_code=map_http_status(envelope.success, code),
        content=envelope.model_dump(),
    )


@router.post(
    "/start",
    response_model=ApiResponse[PlanData],
    summary="Start interview plan from JD",
    dependencies=[Depends(verify_api_key)],
)
def plan_start(body: PlanStartRequestBody):
    response = get_services().plan_service.handle(to_plan_start_request(body))
    return _plan_json_response(to_plan_response(response))


@router.post(
    "/messages",
    response_model=ApiResponse[PlanData],
    summary="Reply to plan clarifying questions",
    dependencies=[Depends(verify_api_key)],
)
def plan_message(body: PlanMessageRequestBody):
    response = get_services().plan_service.handle(to_plan_message_request(body))
    return _plan_json_response(to_plan_response(response))


@router.post(
    "/confirm",
    response_model=ApiResponse[PlanData],
    summary="Confirm interview plan",
    dependencies=[Depends(verify_api_key)],
)
def plan_confirm(body: PlanConfirmRequestBody):
    response = get_services().plan_service.handle(to_plan_confirm_request(body))
    return _plan_json_response(to_plan_response(response))
