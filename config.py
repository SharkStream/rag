import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()


def _normalize_env_name(name: str) -> str:
    aliases = {
        "dev": "dev",
        "development": "dev",
        "prod": "prod",
        "production": "prod",
        "pro": "prod",
    }
    return aliases.get(name, name)


def _load_environment_files() -> None:
    local_env_file = BASE_DIR / ".env"
    if local_env_file.exists():
        load_dotenv(local_env_file, override=False)

    env_name = _normalize_env_name(APP_ENV)
    env_file = BASE_DIR / f".env.{env_name}"
    if env_file.exists():
        load_dotenv(env_file, override=True)


_load_environment_files()


def _default_vector_size() -> int:
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    if provider in {"local", "sentence_transformers", "sentence-transformers"}:
        return 384
    return 1536


class Config:
    APP_ENV = APP_ENV
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("LLM_MODEL", "local-model")

    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "qwen3.7-text-embedding-flash")

    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    LOCAL_EMBEDDING_DEVICE = os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu")

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
    QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", str(_default_vector_size())))

    DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "500"))
    DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))


settings = Config()
