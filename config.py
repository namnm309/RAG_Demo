import os

from dotenv import load_dotenv

load_dotenv()


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return float(value)


CONFIG = {
    "ollama_base_url": _env_str("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    "chat_model": _env_str("CHAT_MODEL", "gemma4b:cloud"),
    "embedding_model": _env_str("EMBEDDING_MODEL", "nomic-embed-text"),
    "data_folder": _env_str("DATA_FOLDER", "data_RAG"),
    "chroma_persist_dir": _env_str("CHROMA_PERSIST_DIR", "./chroma_db"),
    "chroma_system_collection": _env_str("CHROMA_SYSTEM_COLLECTION", "system_kb"),
    "chroma_hr_collection": _env_str("CHROMA_HR_COLLECTION", "hr_kb"),
    "chunk_size": _env_int("CHUNK_SIZE", 1200),
    "chunk_overlap": _env_int("CHUNK_OVERLAP", 200),
    "top_k": _env_int("TOP_K", 5),
    "top_k_system": _env_int("TOP_K_SYSTEM", 5),
    "top_k_hr": _env_int("TOP_K_HR", 5),
    "min_score": _env_float("MIN_SCORE", 0.3),
    "max_history_messages": _env_int("MAX_HISTORY_MESSAGES", 6),
    "max_history_message_chars": _env_int("MAX_HISTORY_MESSAGE_CHARS", 1000),
    "internal_api_key": _env_str("INTERNAL_API_KEY", ""),
}
