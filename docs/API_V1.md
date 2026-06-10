# API v1 Contract

Base URL: `http://localhost:8000/api/v1`

Swagger UI: [http://localhost:8000/api/v1/docs](http://localhost:8000/api/v1/docs)

Auth header (trừ `/health`): `X-Internal-Api-Key: {INTERNAL_API_KEY}`

## Response envelope

Mọi endpoint v1 trả:

```json
{
  "success": true,
  "data": { },
  "error": null,
  "meta": {
    "processing_time_ms": 123.4,
    "session_id": "sess_001",
    "phase": "clarifying",
    "conversation_id": null
  }
}
```

Lỗi:

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "JD_INVALID",
    "message": "...",
    "details": ["..."]
  },
  "meta": { "processing_time_ms": 12 }
}
```

### Error codes

| Code | HTTP |
|------|------|
| `VALIDATION_ERROR` | 400 |
| `UNAUTHORIZED` | 401 |
| `JD_INVALID` | 422 |
| `INGEST_FAILED` | 422 |
| `PLAN_ERROR` | 422 |
| `NOT_READY` | 422 |
| `LLM_ERROR` | 502 |
| `INTERNAL_ERROR` | 500 |

## Endpoints

| Method | Path | Mô tả |
|--------|------|--------|
| GET | `/health` | Health + Ollama |
| GET | `/status` | Chunk counts |
| POST | `/knowledge/system/files` | Upload system docs |
| POST | `/knowledge/hr/{owner_id}/files` | Upload JD |
| POST | `/chat` | RAG Q&A |
| POST | `/interview-plans/start` | Bắt đầu plan |
| POST | `/interview-plans/messages` | Clarify |
| POST | `/interview-plans/confirm` | Xác nhận plan |
| POST | `/interview-questions` | Sinh câu hỏi |

Root probe (không prefix): `GET /health`

## Migration từ API cũ

| Legacy | v1 |
|--------|-----|
| `POST /ingest/system` | `POST /api/v1/knowledge/system/files` |
| `POST /ingest/hr/{id}` | `POST /api/v1/knowledge/hr/{id}/files` |
| `POST /chat` | `POST /api/v1/chat` |
| `POST /generate-plan` action=start | `POST /api/v1/interview-plans/start` |
| `POST /generate-plan` action=message | `POST /api/v1/interview-plans/messages` |
| `POST /generate-plan` action=confirm | `POST /api/v1/interview-plans/confirm` |
| `POST /generate-questions` | `POST /api/v1/interview-questions` |

Legacy endpoints vẫn hoạt động với header `Deprecation: true` (sunset 2026-09-01).

## Luồng HR đầy đủ

```bash
# 1. Ingest
curl -X POST http://localhost:8000/api/v1/knowledge/system/files \
  -H "X-Internal-Api-Key: internal-secret" \
  -F "files=@data_RAG/system/interview_rubric_swe4.md"

curl -X POST http://localhost:8000/api/v1/knowledge/hr/hr_alice/files \
  -H "X-Internal-Api-Key: internal-secret" \
  -F "files=@data_RAG/hr/hr_alice/jd_backend_senior.md"

# 2. Plan
curl -X POST http://localhost:8000/api/v1/interview-plans/start \
  -H "Content-Type: application/json" \
  -H "X-Internal-Api-Key: internal-secret" \
  -d '{"owner_id":"hr_alice","session_id":"sess_001"}'

# 3. Clarify (nếu meta.phase=clarifying)
curl -X POST http://localhost:8000/api/v1/interview-plans/messages \
  -H "Content-Type: application/json" \
  -H "X-Internal-Api-Key: internal-secret" \
  -d '{"owner_id":"hr_alice","session_id":"sess_001","message":"5 câu technical behavioral payments","chat_history":[]}'

# 4. Confirm
curl -X POST http://localhost:8000/api/v1/interview-plans/confirm \
  -H "Content-Type: application/json" \
  -H "X-Internal-Api-Key: internal-secret" \
  -d '{"owner_id":"hr_alice","session_id":"sess_001","plan_draft":{"owner_id":"hr_alice","role":"Backend Engineer","level":"SWE4","question_count":5,"question_types":["technical","behavioral"],"topics":["payments"],"summary":"Plan PV"}}'

# 5. Generate
curl -X POST http://localhost:8000/api/v1/interview-questions \
  -H "Content-Type: application/json" \
  -H "X-Internal-Api-Key: internal-secret" \
  -d '{"confirmed_plan":{"owner_id":"hr_alice","role":"Backend Engineer","level":"SWE4","question_count":5,"question_types":["technical","behavioral"],"topics":["payments"],"summary":"Plan PV"}}'
```

Đọc `success`, `data`, `error.code`, `meta.phase` từ envelope.
