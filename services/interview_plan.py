import json
import re
import time
import uuid
from typing import List, Optional

from openai import OpenAI

from helpers.embeddings import embed_text
from helpers.jd_validator import validate_jd_text
from models.schemas import (
    ChatMessage,
    GeneratePlanRequest,
    GeneratePlanResponse,
    InterviewPlan,
    SourceReference,
)
from services.ingestion import DocumentIngestionService
from services.vector_store import ChromaVectorStore

PLAN_SYSTEM_PROMPT = """Bạn là chuyên gia thiết kế kế hoạch phỏng vấn kỹ thuật.

## Mục tiêu
Phân tích JD và rubric, sau đó:
- Hỏi HR các câu clarify nếu thiếu thông tin quan trọng, HOẶC
- Đề xuất plan tạo câu hỏi phỏng vấn nếu đã đủ thông tin.

## Quy tắc
1. Chỉ dùng thông tin trong [JD/HR] và [HỆ THỐNG]. Không bịa yêu cầu không có trong JD.
2. Trả lời BẰNG TIẾNG VIỆT trong assistant_message.
3. Chỉ trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON.
4. Nếu thiếu bất kỳ: số câu hỏi, loại câu (technical/behavioral/situational), chủ đề kỹ thuật ưu tiên → phase=clarifying.
5. Nếu đủ thông tin hoặc HR đã trả lời đủ trong lịch sử → phase=plan_proposed kèm plan đầy đủ.

## Schema JSON bắt buộc
{
  "phase": "clarifying|plan_proposed",
  "assistant_message": "string",
  "clarifying_questions": ["string"],
  "plan": {
    "owner_id": "string",
    "role": "string",
    "level": "string",
    "question_count": 5,
    "question_types": ["technical", "behavioral"],
    "topics": ["string"],
    "summary": "string",
    "constraints": "string",
    "notes": "string"
  }
}

Khi phase=clarifying: plan có thể null hoặc bỏ qua.
Khi phase=plan_proposed: plan phải đầy đủ các field bắt buộc."""

PLAN_CONFIRM_PROMPT = """Bạn xác nhận kế hoạch phỏng vấn đã được HR duyệt.
Tóm tắt ngắn gọn plan bằng tiếng Việt trong assistant_message.
Chỉ trả về JSON: {"assistant_message": "string"}"""


class InterviewPlanService:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        ingestion_service: DocumentIngestionService,
        client: OpenAI,
        config: dict,
    ):
        self._store = vector_store
        self._ingestion = ingestion_service
        self._client = client
        self._config = config

    def handle(self, request: GeneratePlanRequest) -> GeneratePlanResponse:
        session_id = request.session_id or str(uuid.uuid4())
        if request.action == "start":
            response = self.start_plan(request.owner_id)
        elif request.action == "message":
            response = self.continue_plan(
                request.owner_id,
                request.message,
                request.chat_history,
            )
        elif request.action == "confirm":
            response = self.confirm_plan(
                request.owner_id,
                request.plan_draft,
                request.chat_history,
            )
        else:
            response = GeneratePlanResponse(
                success=False,
                error=f"action không hợp lệ: {request.action}",
            )
        response.session_id = session_id
        return response

    def start_plan(self, owner_id: str) -> GeneratePlanResponse:
        start = time.time()
        owner_id = (owner_id or "").strip()
        if not owner_id:
            return GeneratePlanResponse(success=False, error="owner_id không hợp lệ.")

        if self._store.count_for_owner(owner_id) == 0:
            return GeneratePlanResponse(
                success=False,
                phase="jd_invalid",
                error=f"Chưa có JD cho owner '{owner_id}'. Upload qua ingest-hr trước.",
            )

        combined_jd = self._ingestion.load_combined_hr_jd_text(owner_id)
        if not combined_jd.strip():
            return GeneratePlanResponse(
                success=False,
                phase="jd_invalid",
                error=f"Không đọc được file JD trong data_RAG/hr/{owner_id}/.",
            )

        validation = validate_jd_text(
            combined_jd, f"hr/{owner_id} (tổng hợp)", self._config
        )
        if not validation.valid:
            return GeneratePlanResponse(
                success=False,
                phase="jd_invalid",
                error="JD không đạt chuẩn. Vui lòng upload lại.",
                validation_errors=validation.errors,
                assistant_message=(
                    "JD hiện tại không đủ điều kiện để lập plan. "
                    + " ".join(validation.errors)
                ),
                processing_time_ms=(time.time() - start) * 1000,
            )

        system_chunks, hr_chunks = self._retrieve_plan_context(
            owner_id, "job description requirements role level skills interview focus"
        )
        user_message = self._build_analysis_message(
            owner_id, combined_jd, system_chunks, hr_chunks, chat_history=[]
        )
        return self._call_plan_llm(
            owner_id, user_message, system_chunks, hr_chunks, start
        )

    def continue_plan(
        self,
        owner_id: str,
        message: str,
        chat_history: List[ChatMessage],
    ) -> GeneratePlanResponse:
        start = time.time()
        owner_id = (owner_id or "").strip()
        if not owner_id:
            return GeneratePlanResponse(success=False, error="owner_id không hợp lệ.")
        if not message or not message.strip():
            return GeneratePlanResponse(success=False, error="message không được để trống.")

        max_turns = self._config.get("max_plan_turns", 5)
        user_turns = sum(1 for m in chat_history if m.role == "user")
        if user_turns >= max_turns:
            return GeneratePlanResponse(
                success=False,
                error=f"Đã vượt quá {max_turns} lượt clarify. Vui lòng bắt đầu lại hoặc xác nhận plan.",
            )

        combined_jd = self._ingestion.load_combined_hr_jd_text(owner_id)
        query = f"{message} job description interview plan {owner_id}"
        system_chunks, hr_chunks = self._retrieve_plan_context(owner_id, query)
        user_message = self._build_analysis_message(
            owner_id,
            combined_jd,
            system_chunks,
            hr_chunks,
            chat_history,
            hr_reply=message.strip(),
        )
        return self._call_plan_llm(
            owner_id, user_message, system_chunks, hr_chunks, start
        )

    def confirm_plan(
        self,
        owner_id: str,
        plan_draft: Optional[InterviewPlan],
        chat_history: List[ChatMessage],
    ) -> GeneratePlanResponse:
        start = time.time()
        owner_id = (owner_id or "").strip()
        if not plan_draft:
            return GeneratePlanResponse(success=False, error="plan_draft là bắt buộc khi confirm.")

        plan_error = self._validate_plan_struct(plan_draft, owner_id)
        if plan_error:
            return GeneratePlanResponse(success=False, error=plan_error)

        plan = plan_draft
        plan.owner_id = owner_id

        summary_msg = plan.summary or (
            f"Plan cho {plan.role} ({plan.level}): {plan.question_count} câu, "
            f"loại {', '.join(plan.question_types)}."
        )
        try:
            response = self._client.chat.completions.create(
                model=self._config.get("chat_model", "gemma4b:cloud"),
                messages=[
                    {"role": "system", "content": PLAN_CONFIRM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Plan đã duyệt:\n{json.dumps(_plan_to_dict(plan), ensure_ascii=False)}",
                    },
                ],
                stream=False,
            )
            raw = response.choices[0].message.content or ""
            parsed = _parse_json_object(raw)
            if parsed and parsed.get("assistant_message"):
                summary_msg = str(parsed["assistant_message"]).strip()
        except Exception:
            pass

        return GeneratePlanResponse(
            success=True,
            phase="confirmed",
            assistant_message=summary_msg,
            plan=plan,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def _retrieve_plan_context(self, owner_id: str, query: str):
        min_score = self._config.get("min_score", 0.3)
        embed_model = self._config.get("embedding_model", "nomic-embed-text")
        top_k_system = self._config.get("top_k_system", 5)
        top_k_hr = self._config.get("top_k_hr", 5)

        query_embedding = embed_text(self._client, embed_model, query)
        system_chunks = self._store.search(
            query_embedding,
            collection="system",
            top_k=top_k_system,
            min_score=min_score,
        )
        hr_chunks = self._store.search(
            query_embedding,
            collection="hr",
            top_k=top_k_hr,
            min_score=min_score,
            owner_id=owner_id,
        )
        return system_chunks, hr_chunks

    def _build_analysis_message(
        self,
        owner_id: str,
        combined_jd: str,
        system_chunks,
        hr_chunks,
        chat_history: List[ChatMessage],
        hr_reply: str = "",
    ) -> str:
        lines = [
            f"owner_id: {owner_id}",
            "\n[JD/HR — toàn bộ nội dung đã upload]",
            combined_jd[:12000],
        ]
        lines.append("\n[HỆ THỐNG — rubric]")
        if system_chunks:
            for i, c in enumerate(system_chunks, 1):
                lines.append(
                    f"--- {i} ({c.source_file}, chunk #{c.chunk_index}) ---\n{c.text}"
                )
        else:
            lines.append("(không có rubric system)")

        lines.append("\n[HR — chunks retrieve]")
        if hr_chunks:
            for i, c in enumerate(hr_chunks, 1):
                lines.append(
                    f"--- {i} ({c.source_file}, chunk #{c.chunk_index}) ---\n{c.text}"
                )

        if chat_history:
            lines.append("\n[LỊCH SỬ HỘI THOẠI]")
            for msg in chat_history[-self._config.get("max_history_messages", 6) :]:
                if msg.role in ("user", "assistant") and msg.content.strip():
                    lines.append(f"{msg.role}: {msg.content.strip()}")

        if hr_reply:
            lines.append(f"\n[PHẢN HỒI MỚI CỦA HR]\n{hr_reply}")

        lines.append(
            "\nPhân tích và trả về JSON theo schema. "
            "Nếu thiếu thông tin quan trọng, đặt 2-4 câu hỏi clarify cụ thể."
        )
        return "\n".join(lines)

    def _call_plan_llm(self, owner_id, user_message, system_chunks, hr_chunks, start):
        try:
            response = self._client.chat.completions.create(
                model=self._config.get("chat_model", "gemma4b:cloud"),
                messages=[
                    {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                stream=False,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            return GeneratePlanResponse(
                success=False,
                error=f"Không gọi được LLM: {exc}",
                processing_time_ms=(time.time() - start) * 1000,
            )

        parsed = _parse_json_object(raw)
        if not parsed:
            return GeneratePlanResponse(
                success=False,
                error="LLM không trả về JSON hợp lệ.",
                raw_answer=raw,
                processing_time_ms=(time.time() - start) * 1000,
            )

        phase = str(parsed.get("phase", "clarifying")).strip().lower()
        if phase not in ("clarifying", "plan_proposed"):
            phase = "clarifying"

        assistant_message = str(parsed.get("assistant_message", "")).strip()
        clarifying = parsed.get("clarifying_questions", [])
        if not isinstance(clarifying, list):
            clarifying = []
        clarifying = [str(q).strip() for q in clarifying if str(q).strip()]

        plan = None
        if phase == "plan_proposed":
            plan = _parse_plan_object(parsed.get("plan"), owner_id)
            if not plan:
                return GeneratePlanResponse(
                    success=False,
                    error="JSON plan không hợp lệ.",
                    raw_answer=raw,
                    assistant_message=assistant_message,
                    processing_time_ms=(time.time() - start) * 1000,
                )
            plan_error = self._validate_plan_struct(plan, owner_id)
            if plan_error:
                return GeneratePlanResponse(
                    success=False,
                    error=plan_error,
                    raw_answer=raw,
                    processing_time_ms=(time.time() - start) * 1000,
                )

        sources = _chunks_to_sources(system_chunks + hr_chunks)
        return GeneratePlanResponse(
            success=True,
            phase=phase,  # type: ignore[arg-type]
            assistant_message=assistant_message,
            clarifying_questions=clarifying,
            plan=plan,
            sources=sources,
            raw_answer=raw,
            processing_time_ms=(time.time() - start) * 1000,
        )

    @staticmethod
    def _validate_plan_struct(plan: InterviewPlan, owner_id: str) -> Optional[str]:
        if not plan.role or not plan.role.strip():
            return "Plan thiếu role."
        if not plan.level or not plan.level.strip():
            return "Plan thiếu level."
        if plan.question_count < 1 or plan.question_count > 30:
            return "question_count phải từ 1 đến 30."
        if not plan.question_types:
            return "Plan thiếu question_types."
        valid_types = {"technical", "behavioral", "situational"}
        if not any(t.lower() in valid_types for t in plan.question_types):
            return "question_types phải gồm technical, behavioral hoặc situational."
        if owner_id and plan.owner_id and plan.owner_id != owner_id:
            return "plan.owner_id không khớp request."
        return None


def _parse_json_object(raw: str) -> Optional[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            data = json.loads(match.group())
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def _parse_plan_object(data: object, owner_id: str) -> Optional[InterviewPlan]:
    if not isinstance(data, dict):
        return None
    role = str(data.get("role", "")).strip()
    level = str(data.get("level", "")).strip()
    if not role or not level:
        return None
    try:
        question_count = int(data.get("question_count", 5))
    except (TypeError, ValueError):
        question_count = 5
    types_raw = data.get("question_types", ["technical", "behavioral"])
    if isinstance(types_raw, str):
        question_types = [t.strip() for t in types_raw.split(",") if t.strip()]
    elif isinstance(types_raw, list):
        question_types = [str(t).strip() for t in types_raw if str(t).strip()]
    else:
        question_types = ["technical", "behavioral"]
    topics_raw = data.get("topics", [])
    if isinstance(topics_raw, str):
        topics = [t.strip() for t in topics_raw.split(",") if t.strip()]
    elif isinstance(topics_raw, list):
        topics = [str(t).strip() for t in topics_raw if str(t).strip()]
    else:
        topics = []
    return InterviewPlan(
        owner_id=str(data.get("owner_id", owner_id)).strip() or owner_id,
        role=role,
        level=level,
        question_count=question_count,
        question_types=question_types,
        topics=topics,
        summary=str(data.get("summary", "")).strip(),
        constraints=str(data.get("constraints", "")).strip(),
        notes=str(data.get("notes", "")).strip(),
    )


def _plan_to_dict(plan: InterviewPlan) -> dict:
    return {
        "owner_id": plan.owner_id,
        "role": plan.role,
        "level": plan.level,
        "question_count": plan.question_count,
        "question_types": plan.question_types,
        "topics": plan.topics,
        "summary": plan.summary,
        "constraints": plan.constraints,
        "notes": plan.notes,
    }


def _chunks_to_sources(chunks) -> List[SourceReference]:
    return [
        SourceReference(
            file_name=c.source_file,
            chunk_index=c.chunk_index,
            score=c.score,
            text_preview=c.text[:120] + "..." if len(c.text) > 120 else c.text,
            knowledge_base=c.knowledge_base,
            doc_type=c.doc_type,
        )
        for c in chunks
    ]
