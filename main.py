"""
RAG Interview Question Generator — Console (Ollama + ChromaDB)

Cach chay:
    python main.py

Lenh:
    ingest-system       -> index data_RAG/system -> system_kb
    ingest-hr <id>      -> index data_RAG/hr/<id> -> hr_kb
    generate            -> sinh cau hoi phong van (dual RAG)
    status              -> so chunk moi collection
    clear               -> xoa man hinh
    quit                -> thoat
    <cau hoi>           -> chat RAG (co the them owner: <id> | <cau hoi>)
"""

import json
import logging
import os
import sys

from openai import OpenAI
from tqdm import tqdm

from config import CONFIG
from models.schemas import ChatRequest, GenerateQuestionsRequest
from services.ingestion import DocumentIngestionService
from services.interview_generator import InterviewQuestionService
from services.rag_chat import RagChatService
from services.vector_store import ChromaVectorStore

logging.basicConfig(level=logging.WARNING)


class IngestProgressBar:
    def __init__(self, label="Dang xu ly"):
        self._bar = tqdm(
            total=100,
            desc=label,
            bar_format="{desc:<45} |{bar:40}| {percentage:3.0f}%",
            ascii=" #",
            leave=False,
        )
        self._current = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._bar.close()

    def update(self, message, progress):
        next_value = int(progress * 100)
        if next_value > self._current:
            self._bar.update(next_value - self._current)
            self._current = next_value
        self._bar.set_description_str(f"LOADING... {message[:34]}")


def print_separator():
    print("-" * 60)


def print_sources(sources):
    if not sources:
        return
    print("\nNguon:")
    for s in sources:
        preview = s.text_preview[:100] + "..." if len(s.text_preview) > 100 else s.text_preview
        kb = getattr(s, "knowledge_base", "system")
        print(f"   - [{kb}] {s.file_name} (chunk #{s.chunk_index}, score: {s.score:.3f})")
        print(f"     \"{preview}\"")


def do_ingest_system(ingestion_service: DocumentIngestionService):
    print()
    with IngestProgressBar("LOADING... ingest-system") as loading:
        result = ingestion_service.ingest_system(progress_callback=loading.update)
    _print_ingest_result(result)


def do_ingest_hr(ingestion_service: DocumentIngestionService, owner_id: str):
    print()
    with IngestProgressBar(f"LOADING... ingest-hr {owner_id}") as loading:
        result = ingestion_service.ingest_hr(owner_id, progress_callback=loading.update)
    _print_ingest_result(result)


def _print_ingest_result(result):
    if result.success:
        print(f"OK: {result.message}")
        if result.files:
            print(f"   Files: {', '.join(result.files)}")
    else:
        print(f"Loi: {result.message}")


def print_status(store: ChromaVectorStore):
    print(f"  system_kb: {store.count('system')} chunks")
    print(f"  hr_kb:     {store.count('hr')} chunks")
    sys_files = store.indexed_files("system")
    if sys_files:
        print(f"  System files: {', '.join(sys_files[:5])}{'...' if len(sys_files) > 5 else ''}")


def run_generate_wizard(interview_service: InterviewQuestionService):
    print("\n--- Sinh cau hoi phong van (dual RAG) ---")
    owner_id = input("owner_id (vd hr_alice): ").strip()
    if not owner_id:
        print("Can owner_id.\n")
        return
    role = input("Vai tro (vd Backend Engineer): ").strip() or "Backend Engineer"
    level = input("Level (vd SWE4): ").strip() or "SWE4"
    count_str = input("So cau hoi (mac dinh 5): ").strip() or "5"
    try:
        count = int(count_str)
    except ValueError:
        count = 5
    types_str = input("Loai (technical,behavioral — mac dinh ca hai): ").strip()
    if types_str:
        types = [t.strip() for t in types_str.split(",") if t.strip()]
    else:
        types = ["technical", "behavioral"]
    topic = input("Chu de / ky nang (vd git, system design — Enter de bo qua): ").strip()
    extra = input("Bo sung khac (Enter de bo qua): ").strip()
    extra_context = ", ".join(p for p in (topic, extra) if p)

    print("\nDang retrieve + sinh cau hoi...")
    response = interview_service.generate(
        GenerateQuestionsRequest(
            owner_id=owner_id,
            role=role,
            level=level,
            question_count=count,
            question_types=types,
            extra_context=extra_context,
        )
    )

    print_separator()
    if not response.success:
        print(f"Loi: {response.error or 'Khong tao duoc cau hoi'}")
        if response.raw_answer:
            print("\nRaw LLM:\n", response.raw_answer[:2000])
    else:
        print(f"\nDa tao {len(response.questions)} cau hoi ({response.processing_time_ms:.0f}ms):\n")
        for i, q in enumerate(response.questions, 1):
            print(f"{i}. [{q.question_type}/{q.difficulty}] {q.question}")
            print(f"   Ly do: {q.rationale}")
            if q.sample_answer:
                preview = q.sample_answer[:200] + "..." if len(q.sample_answer) > 200 else q.sample_answer
                print(f"   Tra loi mau: {preview}")
            for j, cit in enumerate(q.citations, 1):
                ex = cit.excerpt[:120] + "..." if len(cit.excerpt) > 120 else cit.excerpt
                print(
                    f"   Trich dan {j} [{cit.knowledge_base}] {cit.source_file} "
                    f"chunk #{cit.chunk_index}: \"{ex}\""
                )
            if q.sources and not q.citations:
                print(f"   Nguon: {', '.join(q.sources)}")
            print()
        print("JSON:")
        print(json.dumps(response.to_json_dict(), ensure_ascii=False, indent=2))
    print_sources(response.sources)
    print_separator()
    print()


def parse_chat_input(user_input: str) -> ChatRequest:
    if "|" in user_input:
        owner_part, question = user_input.split("|", 1)
        owner_id = owner_part.strip().removeprefix("owner:").strip()
        return ChatRequest(question=question.strip(), owner_id=owner_id or None)
    return ChatRequest(question=user_input)


def main():
    client = OpenAI(api_key="ollama", base_url=CONFIG["ollama_base_url"])

    try:
        client.models.list()
    except Exception:
        print("Loi: Khong ket noi duoc Ollama. Chay Ollama tai http://localhost:11434")
        sys.exit(1)

    vector_store = ChromaVectorStore(
        persist_dir=CONFIG["chroma_persist_dir"],
        system_collection=CONFIG["chroma_system_collection"],
        hr_collection=CONFIG["chroma_hr_collection"],
    )
    ingestion_service = DocumentIngestionService(vector_store, client, CONFIG)
    rag_service = RagChatService(vector_store, client, CONFIG)
    interview_service = InterviewQuestionService(vector_store, client, CONFIG)

    print("\n" + "=" * 60)
    print("  Interview RAG  (Ollama + ChromaDB)")
    print("=" * 60)
    print(f"  Model    : {CONFIG['chat_model']}")
    print(f"  Embed    : {CONFIG['embedding_model']}")
    print(f"  Chroma   : {CONFIG['chroma_persist_dir']}")
    print(f"  Data     : {CONFIG['data_folder']}/system | hr/<user_id>")
    print("-" * 60)
    print("  Lenh: ingest-system | ingest-hr <id> | generate | status")
    print("        owner:<id> | <cau hoi>  (chat co HR context)")
    print("        clear | quit")
    print("=" * 60)

    print()
    ingestion_service.check_existing_data()

    print()
    while True:
        try:
            user_input = input("Ban: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTam biet!")
            break

        if not user_input:
            continue

        low = user_input.lower()
        if low in ("quit", "exit", "q"):
            print("Tam biet!")
            break
        if low == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue
        if low == "status":
            print_status(vector_store)
            print()
            continue
        if low == "generate":
            if not vector_store.is_ready:
                print("Chua co du lieu. Chay ingest-system truoc.\n")
                continue
            run_generate_wizard(interview_service)
            continue
        if low == "ingest-system":
            do_ingest_system(ingestion_service)
            print()
            continue
        if low.startswith("ingest-hr"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Dung: ingest-hr <user_id>\n")
                continue
            do_ingest_hr(ingestion_service, parts[1].strip())
            print()
            continue
        if low == "ingest":
            print("Lenh 'ingest' da thay bang 'ingest-system' va 'ingest-hr <user_id>'.\n")
            continue

        if not vector_store.is_ready:
            print("Chua co du lieu. Chay ingest-system truoc.\n")
            continue

        chat_req = parse_chat_input(user_input)
        print("\nDang tim kiem va tra loi...")
        response = rag_service.ask(chat_req)

        print_separator()
        print(f"\nBot:\n{response.answer}")
        print_sources(response.sources)
        print(f"\n{response.processing_time_ms:.0f}ms | {response.chunks_used} chunks")
        print_separator()
        print()


if __name__ == "__main__":
    main()
