from models.api.v1.chat import ChatRequestBody
from models.api.v1.common import ChatMessageBody, InterviewPlanBody
from models.api.v1.plan import PlanConfirmRequestBody, PlanMessageRequestBody, PlanStartRequestBody
from models.api.v1.questions import GenerateQuestionsRequestBody
from models.schemas import (
    ChatMessage,
    ChatRequest,
    GeneratePlanRequest,
    GenerateQuestionsRequest,
    InterviewPlan,
)


def to_chat_messages(messages: list[ChatMessageBody]) -> list[ChatMessage]:
    return [
        ChatMessage(role=m.role, content=m.content, created_at=m.created_at)
        for m in messages
    ]


def to_chat_request(body: ChatRequestBody) -> ChatRequest:
    return ChatRequest(
        question=body.question,
        top_k=body.top_k,
        owner_id=body.owner_id,
        conversation_id=body.conversation_id,
        chat_history=to_chat_messages(body.chat_history),
    )


def to_interview_plan(body: InterviewPlanBody) -> InterviewPlan:
    return InterviewPlan(
        owner_id=body.owner_id,
        role=body.role,
        level=body.level,
        question_count=body.question_count,
        question_types=list(body.question_types),
        topics=list(body.topics),
        summary=body.summary,
        constraints=body.constraints,
        notes=body.notes,
    )


def to_plan_start_request(body: PlanStartRequestBody) -> GeneratePlanRequest:
    return GeneratePlanRequest(
        action="start",
        owner_id=body.owner_id,
        session_id=body.session_id,
        chat_history=[],
    )


def to_plan_message_request(body: PlanMessageRequestBody) -> GeneratePlanRequest:
    return GeneratePlanRequest(
        action="message",
        owner_id=body.owner_id,
        session_id=body.session_id,
        message=body.message,
        chat_history=to_chat_messages(body.chat_history),
    )


def to_plan_confirm_request(body: PlanConfirmRequestBody) -> GeneratePlanRequest:
    return GeneratePlanRequest(
        action="confirm",
        owner_id=body.owner_id,
        session_id=body.session_id,
        plan_draft=to_interview_plan(body.plan_draft),
        chat_history=to_chat_messages(body.chat_history),
    )


def to_generate_questions_request(body: GenerateQuestionsRequestBody) -> GenerateQuestionsRequest:
    return GenerateQuestionsRequest(
        owner_id=body.owner_id,
        role=body.role,
        level=body.level,
        question_count=body.question_count,
        question_types=list(body.question_types),
        extra_context=body.extra_context,
        top_k_system=body.top_k_system,
        top_k_hr=body.top_k_hr,
        confirmed_plan=to_interview_plan(body.confirmed_plan) if body.confirmed_plan else None,
    )
