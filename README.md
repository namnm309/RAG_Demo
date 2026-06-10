# note — Interview RAG

RAG + Ollama + Chroma → sinh câu hỏi PV có **trích dẫn doc** + **đáp án mẫu**.  
`python main.py` (console thôi, chưa web)

---

**2 kb:**
- `system` = rubric / sách chung → `data_RAG/system/`
- `hr` = JD, policy từng người → `data_RAG/hr/hr_alice/` …

---

**chạy lần đầu**
```bash
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull gemma4:31b-cloud
python main.py
```
→ bật Ollama. sửa model ở `config.py` nếu cần

---

**nhớ lệnh**
- `ingest-system` — nạp folder system (xóa system cũ rồi index lại)
- `ingest-hr hr_alice` — nạp JD alice, không đụng bob/system
- `status` — xem có chunk chưa
- `generate` — ra JSON câu hỏi ← **cái chính**
- `owner:hr_alice | ...` — chat thử

thêm/sửa file → ingest lại

---

**generate cho đúng**
1. ingest-system (+ ingest-hr nếu cần JD)
2. `generate` → nhập owner, role, level
3. **chủ đề** (vd `git`) — không nhập thì hay chỉ lấy rubric SWE4
4. cuối có JSON: `citations`, `sample_answer`

⚠ chat ≠ generate. muốn bộ câu hỏi → `generate`, đừng chat "tạo câu hỏi git"

---

**dính lỗi**
- 0 chunk → file sai folder (phải trong `system/` hoặc `hr/<id>/`)
- không thấy pro git → ingest-system + generate + chủ đề `git`
- hr không có jd → `ingest-hr` + owner_id trùng tên folder
- retrieve ít → hạ `min_score` config (0.2)

db: `chroma_db/` | `rag_cache/` = đợt cũ, bỏ qua
