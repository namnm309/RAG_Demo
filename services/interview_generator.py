import json
import re
import time
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from helpers.embeddings import embed_text
from models.schemas import (
    DocumentChunk,
    GeneratedQuestion,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    QuestionCitation,
    SourceReference,
)
from services.vector_store import ChromaVectorStore

EXCERPT_FALLBACK_LEN = 300

INTERVIEW_SYSTEM_PROMPT = """Ban la chuyen gia thiet ke cau hoi phong van ky thuat.

## Muc tieu
Tao bo cau hoi phong van dua tren [HE THONG] (rubric, question bank, tai lieu ky thuat) va [HR] (JD, policy, form).

## Quy tac
1. Chi dung thong tin trong [HE THONG] va [HR]. Khong bia yeu cau khong co trong JD.
2. Moi cau hoi phai bam level va role trong yeu cau.
3. Can bang loai cau hoi theo question_types duoc yeu cau.
4. Tra loi BANG TIENG VIET.
5. Chi tra ve JSON hop le, khong markdown, khong giai thich ngoai JSON.
6. Moi cau hoi phai co it nhat 1 citation neu co tai lieu lien quan trong context.
7. excerpt phai la trich nguyen van ngan tu doan context (khong paraphrase).
8. sample_answer chi duoc suy tu excerpt/citations; neu khong du can cu: "Khong du can cu trong tai lieu."

## Schema JSON bat buoc
{
  "questions": [
    {
      "question": "string",
      "question_type": "technical|behavioral|situational",
      "difficulty": "easy|medium|hard",
      "rationale": "string",
      "sample_answer": "string",
      "citations": [
        {
          "knowledge_base": "system|hr",
          "source_file": "ten_file",
          "chunk_index": 0,
          "excerpt": "doan trich ngan nguyen van"
        }
      ]
    }
  ]
}"""


class InterviewQuestionService:
    def __init__(self, vector_store: ChromaVectorStore, client: OpenAI, config: dict):
        self._store = vector_store
        self._client = client
        self._config = config

    def generate(self, request: GenerateQuestionsRequest) -> GenerateQuestionsResponse:
        if not self._store.is_ready:
            return GenerateQuestionsResponse(
                success=False,
                error="Chua co du lieu. Chay ingest-system va ingest-hr truoc.",
            )

        start = time.time()
        top_k_system = request.top_k_system or self._config.get("top_k_system", 5)
        top_k_hr = request.top_k_hr or self._config.get("top_k_hr", 5)
        min_score = self._config.get("min_score", 0.3)
        embed_model = self._config.get("embedding_model", "nomic-embed-text")

        query = self._build_retrieval_query(request)
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
            owner_id=request.owner_id,
        )

        all_chunks = system_chunks + hr_chunks
        if not all_chunks:
            return GenerateQuestionsResponse(
                success=False,
                error="Khong tim thay tai lieu lien quan trong system_kb hoac hr_kb.",
                processing_time_ms=(time.time() - start) * 1000,
            )

        user_message = self._build_user_message(request, system_chunks, hr_chunks)

        response = self._client.chat.completions.create(
            model=self._config.get("chat_model", "gemma4:31b-cloud"),
            messages=[
                {"role": "system", "content": INTERVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            stream=False,
        )
        raw = response.choices[0].message.content or ""
        chunk_lookup = _build_chunk_lookup(all_chunks)
        questions, parse_error = self._parse_questions(raw, chunk_lookup)

        sources = [
            SourceReference(
                file_name=c.source_file,
                chunk_index=c.chunk_index,
                score=c.score,
                text_preview=c.text[:120] + "..." if len(c.text) > 120 else c.text,
                knowledge_base=c.knowledge_base,
                doc_type=c.doc_type,
            )
            for c in all_chunks
        ]

        return GenerateQuestionsResponse(
            success=parse_error is None and len(questions) > 0,
            questions=questions,
            raw_answer=raw,
            sources=sources,
            processing_time_ms=(time.time() - start) * 1000,
            error=parse_error,
        )

    @staticmethod
    def _build_retrieval_query(request: GenerateQuestionsRequest) -> str:
        types = ", ".join(request.question_types)
        parts = [
            request.level,
            request.role,
            "interview questions rubric",
            types,
            request.extra_context,
        ]
        return " ".join(p for p in parts if p).strip()

    def _build_user_message(self, request: GenerateQuestionsRequest, system_chunks, hr_chunks) -> str:
        lines = [
            "[YEU CAU]",
            f"owner_id: {request.owner_id}",
            f"role: {request.role}",
            f"level: {request.level}",
            f"so_cau: {request.question_count}",
            f"loai_cau: {', '.join(request.question_types)}",
        ]
        if request.extra_context:
            lines.append(f"chu_de_ky_nang: {request.extra_context}")

        lines.append(
            "\nHuong dan: moi citation phai khop dung source_file va chunk_index "
            "trong tung doan duoi. excerpt trich nguyen van tu doan do."
        )

        lines.append("\n[HE THONG]")
        if system_chunks:
            for i, c in enumerate(system_chunks, 1):
                lines.append(
                    f"--- {i} (source_file={c.source_file}, chunk_index={c.chunk_index}, "
                    f"knowledge_base=system, doc_type={c.doc_type}) ---\n{c.text}"
                )
        else:
            lines.append("(khong co doan nao)")

        lines.append("\n[HR]")
        if hr_chunks:
            for i, c in enumerate(hr_chunks, 1):
                lines.append(
                    f"--- {i} (source_file={c.source_file}, chunk_index={c.chunk_index}, "
                    f"knowledge_base=hr, doc_type={c.doc_type}) ---\n{c.text}"
                )
        else:
            lines.append("(khong co JD/policy cho user nay)")

        lines.append(f"\nTao dung {request.question_count} cau hoi. Tra ve JSON theo schema.")
        return "\n".join(lines)

    def _parse_questions(
        self,
        raw: str,
        chunk_lookup: Dict[Tuple[str, str, int], DocumentChunk],
    ) -> tuple[List[GeneratedQuestion], Optional[str]]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return [], "LLM khong tra ve JSON hop le."
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError as e:
                return [], f"Khong parse duoc JSON: {e}"

        items = data.get("questions", data if isinstance(data, list) else [])
        if not isinstance(items, list):
            return [], "JSON thieu mang 'questions'."

        questions: List[GeneratedQuestion] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            q_text = item.get("question", "").strip()
            if not q_text:
                continue

            citations = _parse_citations_from_item(item)
            if not citations:
                citations = _parse_legacy_sources(item.get("sources", []))

            citations = _enrich_citations(citations, chunk_lookup)

            sources = [
                f"{c.source_file}, chunk #{c.chunk_index}"
                for c in citations
            ]

            questions.append(
                GeneratedQuestion(
                    question=q_text,
                    question_type=str(item.get("question_type", "technical")),
                    difficulty=str(item.get("difficulty", "medium")),
                    rationale=str(item.get("rationale", "")),
                    sample_answer=str(item.get("sample_answer", "")).strip(),
                    citations=citations,
                    sources=sources,
                )
            )

        if not questions:
            return [], "JSON khong co cau hoi hop le."
        return questions, None


def _build_chunk_lookup(chunks: List[DocumentChunk]) -> Dict[Tuple[str, str, int], DocumentChunk]:
    lookup: Dict[Tuple[str, str, int], DocumentChunk] = {}
    for c in chunks:
        key = (c.knowledge_base, c.source_file, int(c.chunk_index))
        lookup[key] = c
    return lookup


def _parse_citations_from_item(item: dict) -> List[QuestionCitation]:
    raw_citations = item.get("citations", [])
    if not isinstance(raw_citations, list):
        return []

    result: List[QuestionCitation] = []
    for cit in raw_citations:
        if not isinstance(cit, dict):
            continue
        kb = str(cit.get("knowledge_base", "system")).strip().lower()
        if kb not in ("system", "hr"):
            kb = "system"
        source_file = str(cit.get("source_file", "")).strip()
        if not source_file:
            continue
        try:
            chunk_index = int(cit.get("chunk_index", 0))
        except (TypeError, ValueError):
            chunk_index = 0
        excerpt = str(cit.get("excerpt", "")).strip()
        result.append(
            QuestionCitation(
                knowledge_base=kb,
                source_file=source_file,
                chunk_index=chunk_index,
                excerpt=excerpt,
            )
        )
    return result


def _parse_legacy_sources(sources: object) -> List[QuestionCitation]:
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list):
        return []

    result: List[QuestionCitation] = []
    pattern = re.compile(
        r"(?P<file>[^,]+?)\s*,\s*chunk\s*#?\s*(?P<idx>\d+)",
        re.IGNORECASE,
    )
    for entry in sources:
        text = str(entry).strip()
        match = pattern.search(text)
        if not match:
            continue
        result.append(
            QuestionCitation(
                knowledge_base="system",
                source_file=match.group("file").strip(),
                chunk_index=int(match.group("idx")),
                excerpt="",
            )
        )
    return result


def _enrich_citations(
    citations: List[QuestionCitation],
    chunk_lookup: Dict[Tuple[str, str, int], DocumentChunk],
) -> List[QuestionCitation]:
    enriched: List[QuestionCitation] = []
    for cit in citations:
        key = (cit.knowledge_base, cit.source_file, cit.chunk_index)
        chunk = chunk_lookup.get(key)
        excerpt = cit.excerpt
        if chunk:
            if not excerpt or excerpt not in chunk.text:
                excerpt = _fallback_excerpt(chunk.text)
        elif not excerpt:
            excerpt = ""
        enriched.append(
            QuestionCitation(
                knowledge_base=cit.knowledge_base,
                source_file=cit.source_file,
                chunk_index=cit.chunk_index,
                excerpt=excerpt,
            )
        )
    return enriched


def _fallback_excerpt(chunk_text: str) -> str:
    text = chunk_text.strip()
    if len(text) <= EXCERPT_FALLBACK_LEN:
        return text
    return text[:EXCERPT_FALLBACK_LEN].rstrip() + "..."
