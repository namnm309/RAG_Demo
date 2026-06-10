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
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Hoặc dùng script:

```powershell
.\run-local.ps1
```

## Bước 4 — Test

```powershell
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ingest/system -H "X-Internal-Api-Key: internal-secret"

curl -X POST http://localhost:8000/chat `
  -H "Content-Type: application/json" `
  -H "X-Internal-Api-Key: internal-secret" `
  -d "{\"question\":\"Git rebase là gì?\",\"owner_id\":\"hr_alice\",\"chat_history\":[]}"
```

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
