from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from api.converters import to_chat_request
from api.deps import get_services, verify_api_key
from api.mappers import to_chat_response
from models.api.v1.chat import ChatData, ChatRequestBody
from models.api.v1.common import ApiResponse

router = APIRouter(tags=["Chat"])


@router.post(
    "/chat",
    response_model=ApiResponse[ChatData],
    summary="RAG chat",
    dependencies=[Depends(verify_api_key)],
)
def chat(body: ChatRequestBody):
    response = get_services().rag_service.ask(to_chat_request(body))
    envelope = to_chat_response(response)
    return JSONResponse(status_code=200, content=envelope.model_dump())
