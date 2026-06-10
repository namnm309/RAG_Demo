# Deploy RAG Service trên PC (Docker + Cloudflare Tunnel)

Hướng dẫn host RAG service trên PC, expose ra Internet để Backend ASP.NET Core trên Azure gọi API.

Ollama chạy trực tiếp trên PC host (không trong container). Docker container gọi Ollama qua `host.docker.internal`.

---

## A. Chạy Ollama trên PC

```bash
ollama serve
ollama pull nomic-embed-text
ollama pull gemma4b:cloud
```

---

## B. Tạo file .env

```bash
copy .env.example .env
```

Chỉnh sửa `.env` nếu cần. Với Docker, giữ `OLLAMA_BASE_URL=http://host.docker.internal:11434/v1`.

Chạy FastAPI local (không Docker): đổi thành `OLLAMA_BASE_URL=http://localhost:11434/v1`.

**Không commit file `.env` thật.**

---

## C. Build và chạy Docker

```bash
docker compose up -d --build
```

---

## D. Xem logs

```bash
docker logs -f iqgs-rag-service
```

---

## E. Test health

```bash
curl http://localhost:8000/health
```

---

## F. Ingest system

`multipart/form-data`, field `files` (1+ file). Incremental: chỉ thay chunk của từng file gửi lên.

```bash
curl -X POST http://localhost:8000/ingest/system ^
  -H "X-Internal-Api-Key: internal-secret" ^
  -F "files=@data_RAG/system/interview_rubric_swe4.md" ^
  -F "files=@data_RAG/system/question_bank_behavioral.json"
```

**Postman:** Body → form-data → key `files` (type File).

---

## G. Ingest HR

```bash
curl -X POST http://localhost:8000/ingest/hr/hr_alice ^
  -H "X-Internal-Api-Key: internal-secret" ^
  -F "files=@data_RAG/hr/hr_alice/jd_backend_senior.md" ^
  -F "files=@data_RAG/hr/hr_alice/policy_interview_process.md"
```

File gốc lưu tại `data_RAG/hr/{owner_id}/`. Upload lại cùng tên file = cập nhật chunk file đó.

---

## H. Chat

```bash
curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -H "X-Internal-Api-Key: internal-secret" ^
  -d "{\"question\":\"Git rebase là gì?\",\"owner_id\":\"hr_alice\",\"conversation_id\":\"conv_001\",\"top_k\":5,\"chat_history\":[]}"
```

---

## I. Chat có history

```bash
curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -H "X-Internal-Api-Key: internal-secret" ^
  -d "{\"question\":\"Giải thích thêm ý đó bằng ví dụ thực tế\",\"owner_id\":\"hr_alice\",\"conversation_id\":\"conv_001\",\"top_k\":5,\"chat_history\":[{\"role\":\"user\",\"content\":\"Git rebase là gì?\"},{\"role\":\"assistant\",\"content\":\"Git rebase là thao tác áp dụng lại commit lên một base commit mới.\"}]}"
```

---

## J. Generate plan + questions (luồng chính)

```bash
# 1. Bắt đầu lập plan từ JD
curl -X POST http://localhost:8000/generate-plan ^
  -H "Content-Type: application/json" ^
  -H "X-Internal-Api-Key: internal-secret" ^
  -d "{\"action\":\"start\",\"owner_id\":\"hr_alice\",\"session_id\":\"sess_001\",\"chat_history\":[]}"

# 2. HR trả lời clarify (lặp đến khi phase=plan_proposed)
curl -X POST http://localhost:8000/generate-plan ^
  -H "Content-Type: application/json" ^
  -H "X-Internal-Api-Key: internal-secret" ^
  -d "{\"action\":\"message\",\"owner_id\":\"hr_alice\",\"session_id\":\"sess_001\",\"message\":\"5 câu, technical và behavioral, tập trung payments\",\"chat_history\":[]}"

# 3. Xác nhận plan
curl -X POST http://localhost:8000/generate-plan ^
  -H "Content-Type: application/json" ^
  -H "X-Internal-Api-Key: internal-secret" ^
  -d "{\"action\":\"confirm\",\"owner_id\":\"hr_alice\",\"session_id\":\"sess_001\",\"plan_draft\":{\"owner_id\":\"hr_alice\",\"role\":\"Backend Engineer\",\"level\":\"SWE4\",\"question_count\":5,\"question_types\":[\"technical\",\"behavioral\"],\"topics\":[\"payments\"],\"summary\":\"Plan PV\"},\"chat_history\":[]}"

# 4. Sinh câu hỏi
curl -X POST http://localhost:8000/generate-questions ^
  -H "Content-Type: application/json" ^
  -H "X-Internal-Api-Key: internal-secret" ^
  -d "{\"confirmed_plan\":{\"owner_id\":\"hr_alice\",\"role\":\"Backend Engineer\",\"level\":\"SWE4\",\"question_count\":5,\"question_types\":[\"technical\",\"behavioral\"],\"topics\":[\"payments\"],\"summary\":\"Plan PV\"}}"
```

**One-shot (legacy):**

```bash
curl -X POST http://localhost:8000/generate-questions ^
  -H "Content-Type: application/json" ^
  -H "X-Internal-Api-Key: internal-secret" ^
  -d "{\"owner_id\":\"hr_alice\",\"role\":\"Backend Engineer\",\"level\":\"SWE4\",\"question_count\":5,\"question_types\":[\"technical\",\"behavioral\"],\"extra_context\":\"git, system design\"}"
```

**JD validate:** ingest HR từ chối file quá ngắn/dài hoặc thiếu vai trò/yêu cầu (`JD_MIN_CHARS`, `JD_MAX_CHARS`, ...).

---

## K. Expose bằng Cloudflare Tunnel

```bash
cloudflared tunnel --url http://localhost:8000
```

Cloudflare sẽ in URL dạng `https://xxxxx.trycloudflare.com`.

Backend Azure config:

```
RagService__BaseUrl=https://xxxxx.trycloudflare.com
RagService__ApiKey=internal-secret
RagService__TimeoutSeconds=120
```

Backend gọi RAG với header `X-Internal-Api-Key`. RAG service **không** lưu chat history — Backend tự quản lý Redis/PostgreSQL và truyền `chat_history` trong body.

---

## Ba cách chạy

| Mode | Command |
|------|---------|
| Console dev/test | `python main.py` |
| FastAPI local | `uvicorn api:app --host 0.0.0.0 --port 8000` |
| Docker trên PC | `docker compose up -d --build` |

---

## API endpoints

| Method | Path | Auth | Mô tả |
|--------|------|------|-------|
| GET | `/health` | Không | Health check + Ollama ping |
| GET | `/status` | API key | Chunk counts + indexed files |
| POST | `/chat` | API key | RAG chat |
| POST | `/generate-plan` | API key | Lập plan PV (start / message / confirm) |
| POST | `/generate-questions` | API key | Sinh câu hỏi (dùng `confirmed_plan` hoặc one-shot) |
| POST | `/ingest/system` | API key | Upload file multipart (`files`) → index system KB (incremental) |
| POST | `/ingest/hr/{owner_id}` | API key | Upload file multipart (`files`) → index HR KB theo owner (incremental) |

Auth header: `X-Internal-Api-Key: {INTERNAL_API_KEY}` (bỏ qua nếu `INTERNAL_API_KEY` rỗng).
