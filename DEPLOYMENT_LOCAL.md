# Chạy RAG local (không Docker)

## Yêu cầu

- Python 3.11+
- Ollama đã cài và chạy trên PC

## Bước 1 — Ollama

```powershell
ollama pull nomic-embed-text
ollama pull gemma4:31b-cloud
```

> Model `gemma4b:cloud` không tồn tại trên Ollama registry — dùng `gemma4:31b-cloud` (đã cấu hình trong `.env`).

## Bước 2 — .env

File `.env` đã cấu hình cho local:

- `OLLAMA_BASE_URL=http://localhost:11434/v1`
- `DATA_FOLDER=data_RAG`
- `CHROMA_PERSIST_DIR=./chroma_db`

## Bước 3 — Cài dependency + chạy API

```powershell
cd E:\Github\RagChatbotPython
pip install -r requirements.txt
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Hoặc dùng script:

```powershell
.\run-local.ps1
```

## Bước 4 — Test

**Swagger UI:** http://localhost:8000/api/v1/docs

**API v1** (khuyến nghị): prefix `/api/v1`, response envelope `{ success, data, error, meta }`. Chi tiết: [docs/API_V1.md](docs/API_V1.md).

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/health

# Ingest system — multipart, field "files" (có thể gửi nhiều file)
curl -X POST http://localhost:8000/ingest/system `
  -H "X-Internal-Api-Key: internal-secret" `
  -F "files=@data_RAG/system/interview_rubric_swe4.md" `
  -F "files=@data_RAG/system/question_bank_behavioral.json"

# Ingest HR — incremental theo owner_id
curl -X POST http://localhost:8000/ingest/hr/hr_alice `
  -H "X-Internal-Api-Key: internal-secret" `
  -F "files=@data_RAG/hr/hr_alice/jd_backend_senior.md"

curl -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -H "X-Internal-Api-Key: internal-secret" `
  -d "{\"question\":\"Git rebase là gì?\",\"owner_id\":\"hr_alice\",\"chat_history\":[]}"

# Generate plan — bắt đầu phân tích JD
curl -X POST http://localhost:8000/generate-plan `
  -H "Content-Type: application/json" `
  -H "X-Internal-Api-Key: internal-secret" `
  -d "{\"action\":\"start\",\"owner_id\":\"hr_alice\",\"session_id\":\"sess_001\",\"chat_history\":[]}"

# Generate plan — HR trả lời clarify
curl -X POST http://localhost:8000/generate-plan `
  -H "Content-Type: application/json" `
  -H "X-Internal-Api-Key: internal-secret" `
  -d "{\"action\":\"message\",\"owner_id\":\"hr_alice\",\"session_id\":\"sess_001\",\"message\":\"5 câu, 3 technical 2 behavioral, payments PostgreSQL\",\"chat_history\":[{\"role\":\"assistant\",\"content\":\"...\"},{\"role\":\"user\",\"content\":\"5 câu...\"}]}"

# Generate plan — xác nhận plan (plan_draft từ bước trước)
curl -X POST http://localhost:8000/generate-plan `
  -H "Content-Type: application/json" `
  -H "X-Internal-Api-Key: internal-secret" `
  -d "{\"action\":\"confirm\",\"owner_id\":\"hr_alice\",\"session_id\":\"sess_001\",\"plan_draft\":{\"owner_id\":\"hr_alice\",\"role\":\"Backend Engineer\",\"level\":\"SWE4\",\"question_count\":5,\"question_types\":[\"technical\",\"behavioral\"],\"topics\":[\"payments\",\"PostgreSQL\"],\"summary\":\"Plan PV backend payments\"},\"chat_history\":[]}"

# Sinh câu hỏi từ plan đã xác nhận
curl -X POST http://localhost:8000/generate-questions `
  -H "Content-Type: application/json" `
  -H "X-Internal-Api-Key: internal-secret" `
  -d "{\"confirmed_plan\":{\"owner_id\":\"hr_alice\",\"role\":\"Backend Engineer\",\"level\":\"SWE4\",\"question_count\":5,\"question_types\":[\"technical\",\"behavioral\"],\"topics\":[\"payments\"],\"summary\":\"Plan PV backend\"}}"
```

**Postman ingest:** Body → form-data → key `files` (type File), thêm nhiều row cùng key để upload nhiều file.

**Validate JD (chỉ ingest HR):** file quá ngắn/dài hoặc thiếu nội dung cốt lõi sẽ bị từ chối, không index. Ngưỡng: `JD_MIN_CHARS` (400), `JD_MAX_CHARS` (30000), `JD_MIN_WORDS` (80), `JD_MAX_WORDS` (5000).

**Hành vi incremental:** mỗi request chỉ cập nhật chunk của file được gửi; file khác trong Chroma không bị xóa. File gốc được lưu vào `data_RAG/system/` hoặc `data_RAG/hr/{owner_id}/`.

Định dạng hỗ trợ: `.pdf`, `.docx`, `.txt`, `.md`, `.json`, `.xlsx`, `.xls`. Giới hạn mặc định: 20 file/request, 25 MB/file (`MAX_UPLOAD_FILES`, `MAX_UPLOAD_SIZE_MB` trong `.env`).

## Expose cho Azure (Cloudflare Tunnel)

Terminal riêng (giữ uvicorn đang chạy):

```powershell
cloudflared tunnel --url http://localhost:8000
```

Azure config:

```
RagService__BaseUrl=https://xxxxx.trycloudflare.com
RagService__ApiKey=internal-secret
RagService__TimeoutSeconds=120
```

## Docker

Khi cài Docker Desktop, đổi `.env` về Docker paths hoặc copy từ `.env.example`, rồi `docker compose up -d --build`. Xem [DEPLOYMENT_PC.md](DEPLOYMENT_PC.md).
