from typing import Optional

from models.api.v1.chat import ChatData
from models.api.v1.common import ApiResponse, ErrorBody, ErrorCode, InterviewPlanBody, Meta, SourceRef
from models.api.v1.health import HealthData, StatusData
from models.api.v1.ingest import IngestData, IngestFileResultData
from models.api.v1.plan import PlanData
from models.api.v1.questions import GeneratedQuestionData, QuestionCitationData, QuestionsData
from models.schemas import (
    ChatResponse,
    GeneratePlanResponse,
    GenerateQuestionsResponse,
    IngestResponse,
    InterviewPlan,
    SourceReference,
)


def map_http_status(success: bool, error_code: Optional[ErrorCode]) -> int:
    if success:
        return 200
    if error_code is None:
        return 500
    mapping = {
        ErrorCode.VALIDATION_ERROR: 400,
        ErrorCode.UNAUTHORIZED: 401,
        ErrorCode.JD_INVALID: 422,
        ErrorCode.INGEST_FAILED: 422,
        ErrorCode.PLAN_ERROR: 422,
        ErrorCode.NOT_READY: 422,
        ErrorCode.LLM_ERROR: 502,
        ErrorCode.INTERNAL_ERROR: 500,
    }
    return mapping.get(error_code, 422)


def _source_ref(source: SourceReference) -> SourceRef:
    return SourceRef(
        file_name=source.file_name,
        chunk_index=source.chunk_index,
        score=source.score,
        text_preview=source.text_preview,
        knowledge_base=source.knowledge_base,
        doc_type=source.doc_type,
    )


def _plan_body(plan: Optional[InterviewPlan]) -> Optional[InterviewPlanBody]:
    if not plan:
        return None
    return InterviewPlanBody(
        owner_id=plan.owner_id,
        role=plan.role,
        level=plan.level,
        question_count=plan.question_count,
        question_types=list(plan.question_types),
        topics=list(plan.topics),
        summary=plan.summary,
        constraints=plan.constraints,
        notes=plan.notes,
    )


def to_health_response(
    status: str,
    chat_model: str,
    embedding_model: str,
    ollama_base_url: str,
    system_chunks: int,
    hr_chunks: int,
    message: Optional[str] = None,
) -> ApiResponse[HealthData]:
    data = HealthData(
        status=status,  # type: ignore[arg-type]
        chat_model=chat_model,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        system_chunks=system_chunks,
        hr_chunks=hr_chunks,
        message=message,
    )
    success = status == "ok"
    error = None
    if not success:
        error = ErrorBody(
            code=ErrorCode.LLM_ERROR,
            message=message or "Cannot connect to Ollama",
        )
    return ApiResponse(success=success, data=data, error=error)


def to_status_response(
    system_chunks: int,
    hr_chunks: int,
    system_files: list[str],
    hr_files: list[str],
) -> ApiResponse[StatusData]:
    return ApiResponse(
        success=True,
        data=StatusData(
            system_chunks=system_chunks,
            hr_chunks=hr_chunks,
            system_files=system_files,
            hr_files=hr_files,
        ),
    )


def to_ingest_response(domain: IngestResponse) -> ApiResponse[IngestData]:
    data = IngestData(
        message=domain.message,
        documents_loaded=domain.documents_loaded,
        chunks_created=domain.chunks_created,
        files=list(domain.files),
        failed_files=list(domain.failed_files),
        file_results=[
            IngestFileResultData(
                file_name=r.file_name,
                success=r.success,
                chunks_created=r.chunks_created,
                message=r.message,
                validation_errors=list(r.validation_errors),
            )
            for r in domain.file_results
        ],
    )
    if domain.success:
        return ApiResponse(success=True, data=data)

    has_jd_validation = any(
        r.validation_errors for r in domain.file_results if not r.success
    )
    code = ErrorCode.JD_INVALID if has_jd_validation else ErrorCode.INGEST_FAILED
    details = []
    for r in domain.file_results:
        if not r.success:
            details.extend(r.validation_errors or [r.message])
    return ApiResponse(
        success=False,
        data=data,
        error=ErrorBody(code=code, message=domain.message, details=details),
    )


def to_chat_response(domain: ChatResponse) -> ApiResponse[ChatData]:
    data = ChatData(
        answer=domain.answer,
        sources=[_source_ref(s) for s in domain.sources],
        chunks_used=domain.chunks_used,
    )
    meta = Meta(
        processing_time_ms=domain.processing_time_ms,
        conversation_id=domain.conversation_id,
    )
    return ApiResponse(success=True, data=data, meta=meta)


def to_plan_response(domain: GeneratePlanResponse) -> ApiResponse[PlanData]:
    data = PlanData(
        assistant_message=domain.assistant_message,
        clarifying_questions=list(domain.clarifying_questions),
        plan=_plan_body(domain.plan),
        validation_errors=list(domain.validation_errors),
        sources=[_source_ref(s) for s in domain.sources],
        raw_answer=domain.raw_answer,
    )
    meta = Meta(
        processing_time_ms=domain.processing_time_ms,
        session_id=domain.session_id,
        phase=domain.phase,
    )
    if domain.success:
        return ApiResponse(success=True, data=data, meta=meta)

    code = ErrorCode.JD_INVALID if domain.phase == "jd_invalid" else ErrorCode.PLAN_ERROR
    if domain.error and "LLM" in domain.error:
        code = ErrorCode.LLM_ERROR
    return ApiResponse(
        success=False,
        data=data,
        error=ErrorBody(
            code=code,
            message=domain.error or "Plan request failed",
            details=list(domain.validation_errors),
        ),
        meta=meta,
    )


def to_questions_response(domain: GenerateQuestionsResponse) -> ApiResponse[QuestionsData]:
    data = QuestionsData(
        questions=[
            GeneratedQuestionData(
                question=q.question,
                question_type=q.question_type,
                difficulty=q.difficulty,
                rationale=q.rationale,
                sample_answer=q.sample_answer,
                citations=[
                    QuestionCitationData(
                        knowledge_base=c.knowledge_base,
                        source_file=c.source_file,
                        chunk_index=c.chunk_index,
                        excerpt=c.excerpt,
                    )
                    for c in q.citations
                ],
                sources=list(q.sources),
            )
            for q in domain.questions
        ],
        sources=[_source_ref(s) for s in domain.sources],
        raw_answer=domain.raw_answer,
    )
    meta = Meta(processing_time_ms=domain.processing_time_ms)
    if domain.success:
        return ApiResponse(success=True, data=data, meta=meta)

    code = ErrorCode.NOT_READY if domain.error and "Chưa có dữ liệu" in domain.error else ErrorCode.PLAN_ERROR
    if domain.error and "LLM" in (domain.error or ""):
        code = ErrorCode.LLM_ERROR
    return ApiResponse(
        success=False,
        data=data,
        error=ErrorBody(code=code, message=domain.error or "Generate questions failed"),
        meta=meta,
    )
