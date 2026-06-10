import json
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional

import pdfplumber
from docx import Document as DocxDocument
from openai import OpenAI

from helpers.doc_type import infer_doc_type
from helpers.embeddings import embed_text
from helpers.text_chunker import split_text
from models.schemas import CollectionName, DocumentChunk, IngestResponse
from services.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".json", ".xlsx", ".xls"}


class DocumentIngestionService:
    def __init__(self, vector_store: ChromaVectorStore, client: OpenAI, config: dict):
        self._store = vector_store
        self._client = client
        self._config = config

    def ingest_system(
        self,
        folder_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> IngestResponse:
        base = self._config.get("data_folder", "data_RAG")
        folder = folder_path or os.path.join(os.getcwd(), base, "system")
        return self._ingest_collection(
            folder=folder,
            collection="system",
            knowledge_base="system",
            owner_id=None,
            progress_callback=progress_callback,
            clear_message="system_kb",
        )

    def ingest_hr(
        self,
        owner_id: str,
        folder_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> IngestResponse:
        if not owner_id or not owner_id.strip():
            return IngestResponse(success=False, message="owner_id không hợp lệ.")
        owner_id = owner_id.strip()
        base = self._config.get("data_folder", "data_RAG")
        folder = folder_path or os.path.join(os.getcwd(), base, "hr", owner_id)
        return self._ingest_collection(
            folder=folder,
            collection="hr",
            knowledge_base="hr",
            owner_id=owner_id,
            progress_callback=progress_callback,
            clear_message=f"hr_kb (owner={owner_id})",
        )

    def check_existing_data(self) -> int:
        system_count = self._store.count("system")
        hr_count = self._store.count("hr")
        total = system_count + hr_count
        if total > 0:
            print(f"ChromaDB: system_kb={system_count} chunks, hr_kb={hr_count} chunks.")
        else:
            print("Chưa có dữ liệu. Dùng 'ingest-system' hoặc 'ingest-hr <user_id>'.")
        return total

    def _ingest_collection(
        self,
        folder: str,
        collection: CollectionName,
        knowledge_base: str,
        owner_id: Optional[str],
        progress_callback: Optional[Callable[[str, float], None]],
        clear_message: str,
    ) -> IngestResponse:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            return IngestResponse(
                success=False,
                message=f"Đã tạo folder '{folder}'. Cho file vào và gọi lại.",
            )

        chunk_size = self._config.get("chunk_size", 1200)
        overlap = self._config.get("chunk_overlap", 200)
        embed_model = self._config.get("embedding_model", "nomic-embed-text")

        self._emit_progress(progress_callback, f"Xóa dữ liệu cũ trong {clear_message}...", 0.0)
        deleted = self._store.delete_scoped(collection, owner_id=owner_id)
        self._emit_progress(progress_callback, f"Đã xóa {deleted} bản ghi cũ.", 0.02)

        files = [
            str(p)
            for p in Path(folder).rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            return IngestResponse(success=False, message=f"Không tìm thấy file nào trong '{folder}'.")

        self._emit_progress(progress_callback, f"Tìm thấy {len(files)} file, bắt đầu ingestion...", 0.05)
        total_chunks = 0

        for file_index, file_path in enumerate(files, start=1):
            try:
                fname = os.path.basename(file_path)
                base_progress = (file_index - 1) / len(files)
                self._emit_progress(
                    progress_callback,
                    f"[{file_index}/{len(files)}] Đang đọc: {fname}",
                    base_progress,
                )

                raw_text = self._extract_text(file_path)
                if not raw_text or not raw_text.strip():
                    continue

                chunks = split_text(raw_text, chunk_size, overlap)
                if not chunks:
                    continue

                doc_type = infer_doc_type(file_path, knowledge_base)
                stem = Path(file_path).stem
                batch: List[DocumentChunk] = []

                for chunk_index, chunk_text in enumerate(chunks, start=1):
                    progress = ((file_index - 1) + (chunk_index / len(chunks))) / len(files)
                    self._emit_progress(
                        progress_callback,
                        f"[{file_index}/{len(files)}] Embedding {fname}: chunk {chunk_index}/{len(chunks)}",
                        progress,
                    )
                    cid = (
                        f"{owner_id}_{stem}_chunk_{chunk_index - 1}"
                        if owner_id
                        else f"{stem}_chunk_{chunk_index - 1}"
                    )
                    batch.append(
                        DocumentChunk(
                            id=cid,
                            text=chunk_text,
                            source_file=fname,
                            chunk_index=chunk_index - 1,
                            embedding=embed_text(self._client, embed_model, chunk_text),
                            knowledge_base=knowledge_base,
                            owner_id=owner_id,
                            doc_type=doc_type,
                        )
                    )

                self._store.add_chunks(batch, collection)
                total_chunks += len(batch)

            except Exception as e:
                logger.exception("Ingest error %s", file_path)
                self._emit_progress(
                    progress_callback,
                    f"[!] Lỗi: {os.path.basename(file_path)}: {e}",
                    file_index / len(files),
                )

        self._emit_progress(progress_callback, f"Xong. {total_chunks} chunks trong {clear_message}.", 1.0)
        return IngestResponse(
            success=True,
            message=f"Index thành công {len(files)} file với {total_chunks} chunks ({clear_message}).",
            documents_loaded=len(files),
            chunks_created=total_chunks,
            files=[os.path.basename(f) for f in files],
        )

    def _extract_text(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self._extract_pdf(file_path)
        if ext == ".docx":
            return self._extract_docx(file_path)
        if ext in (".txt", ".md"):
            return Path(file_path).read_text(encoding="utf-8")
        if ext == ".json":
            return self._extract_json(file_path)
        if ext == ".xlsx":
            return self._extract_xlsx(file_path)
        if ext == ".xls":
            return self._extract_xls(file_path)
        return ""

    def _extract_pdf(self, file_path: str) -> str:
        lines = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    lines.append(text)
        return "\n\n".join(lines)

    def _extract_docx(self, file_path: str) -> str:
        doc = DocxDocument(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _extract_json(self, file_path: str) -> str:
        raw = Path(file_path).read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        lines: List[str] = []
        self._flatten_json(data, "", lines)
        return "\n".join(lines)

    def _extract_xlsx(self, file_path: str) -> str:
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise RuntimeError("Cần cài openpyxl. Chạy: pip install -r requirements.txt") from exc

        workbook = load_workbook(file_path, data_only=True, read_only=True)
        lines: List[str] = []
        for sheet in workbook.worksheets:
            lines.append(f"Sheet: {sheet.title}")
            for row in sheet.iter_rows(values_only=True):
                row_text = self._format_excel_row(row)
                if row_text:
                    lines.append(row_text)
            lines.append("")
        workbook.close()
        return "\n".join(lines)

    def _extract_xls(self, file_path: str) -> str:
        try:
            import xlrd
        except ImportError as exc:
            raise RuntimeError("Cần cài xlrd. Chạy: pip install -r requirements.txt") from exc

        workbook = xlrd.open_workbook(file_path)
        lines: List[str] = []
        for sheet in workbook.sheets():
            lines.append(f"Sheet: {sheet.name}")
            for row_index in range(sheet.nrows):
                row_text = self._format_excel_row(sheet.row_values(row_index))
                if row_text:
                    lines.append(row_text)
            lines.append("")
        return "\n".join(lines)

    def _format_excel_row(self, row) -> str:
        values = []
        for cell in row:
            if cell is None:
                continue
            text = str(cell).strip()
            if text:
                values.append(text)
        return " | ".join(values)

    def _flatten_json(self, obj, prefix: str, lines: List[str]) -> None:
        if isinstance(obj, list):
            for item in obj:
                self._flatten_json(item, prefix, lines)
                lines.append("")
        elif isinstance(obj, dict):
            for key, value in obj.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    self._flatten_json(value, full_key, lines)
                else:
                    lines.append(f"{full_key}: {value}")
            lines.append("")
        else:
            lines.append(f"{prefix}: {obj}" if prefix else str(obj))

    def _emit_progress(
        self,
        progress_callback: Optional[Callable[[str, float], None]],
        message: str,
        progress: float,
    ) -> None:
        if progress_callback:
            progress_callback(message, max(0.0, min(progress, 1.0)))
        else:
            print(message)
