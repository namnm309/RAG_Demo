# note — Interview RAG

RAG + Ollama + Chroma → sinh câu hỏi PV có **trích dẫn doc** + **đáp án mẫu**.

---

## Ba cách chạy

| Mode | Command |
|------|---------|
| Console dev/test | `python main.py` |
| FastAPI local | `.\run-local.ps1` hoặc `python -m uvicorn api:app --host 0.0.0.0 --port 8000` |
| Docker trên PC | `docker compose up -d --build` (cần Docker Desktop) |

Chạy local không Docker: [DEPLOYMENT_LOCAL.md](DEPLOYMENT_LOCAL.md).  
Deploy Docker + Cloudflare Tunnel: [DEPLOYMENT_PC.md](DEPLOYMENT_PC.md).

Config qua `.env` (copy từ `.env.example`). Không commit `.env` thật.

---

**2 kb:**
- `system` = rubric / sách chung → `data_RAG/system/`
- `hr` = JD, policy từng người → `data_RAG/hr/hr_alice/` …

---

**chạy lần đầu**
```bash
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull gemma4b:cloud
python main.py
```
→ bật Ollama. sửa model trong `.env` hoặc `config.py` defaults nếu cần

---

**nhớ lệnh**
- `ingest-system` — nạp folder system (xóa system cũ rồi index lại)
- `ingest-hr hr_alice` — nạp JD alice (có validate độ dài + nội dung)
- `status` — xem có chunk chưa
- `generate-plan` — JD → clarify → plan → xác nhận HR → sinh câu hỏi ← **luồng chính**
- `generate` — sinh một bước (dev/test)
- `owner:hr_alice | ...` — chat thử

thêm/sửa file → ingest lại

---

**generate-plan cho đúng**
1. `ingest-system` + `ingest-hr <owner_id>` (JD phải đủ dài, có vai trò + yêu cầu)
2. `generate-plan` → hệ thống hỏi clarify nếu thiếu thông tin
3. xác nhận plan (`y` hoặc `edit`)
4. nhận JSON câu hỏi: `citations`, `sample_answer`

API: `POST /generate-plan` (action: `start` | `message` | `confirm`) rồi `POST /generate-questions` với `confirmed_plan`.

⚠ chat ≠ generate. muốn bộ câu hỏi → `generate-plan`, đừng chat "tạo câu hỏi git"

---

**dính lỗi**
- 0 chunk → file sai folder (phải trong `system/` hoặc `hr/<id>/`)
- JD bị reject khi ingest → xem `validation_errors`; chỉnh `JD_MIN_CHARS` / `JD_MAX_CHARS` trong `.env` nếu cần
- `phase=jd_invalid` khi generate-plan → upload lại JD đạt chuẩn
- không thấy pro git → ingest-system + generate-plan + chủ đề trong plan
- hr không có jd → `ingest-hr` + owner_id trùng tên folder
- retrieve ít → hạ `min_score` config (0.2)

db: `chroma_db/` | `rag_cache/` = đợt cũ, bỏ qua
