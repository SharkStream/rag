from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

warnings.filterwarnings(
    "ignore",
    message=r".*Api key is used with an insecure connection.*",
    category=UserWarning,
)

try:
    import redis
except Exception as exc:  # pragma: no cover
    redis = None
    REDIS_IMPORT_ERROR = exc
else:
    REDIS_IMPORT_ERROR = None

try:
    from qdrant_client import QdrantClient
except Exception as exc:  # pragma: no cover
    QdrantClient = None
    QDRANT_IMPORT_ERROR = exc
else:
    QDRANT_IMPORT_ERROR = None

try:
    from dashscope import TextReRank
except Exception as exc:  # pragma: no cover
    TextReRank = None
    DASHSCOPE_RERANK_IMPORT_ERROR = exc
else:
    DASHSCOPE_RERANK_IMPORT_ERROR = None


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _report_ok(name: str, detail: str) -> None:
    print(f"[OK] {name}: {detail}")


def _report_fail(name: str, detail: str) -> None:
    print(f"[FAIL] {name}: {detail}")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_env_file() -> None:
    project_root = _project_root()
    app_env = os.getenv("APP_ENV", "development").strip().lower()
    env_name = {"dev": "dev", "development": "dev", "prod": "prod", "production": "prod"}.get(app_env, app_env)

    env_files = [project_root / f".env.{env_name}", project_root / ".env"]
    for env_file in env_files:
        if env_file.exists():
            load_dotenv(env_file, override=False)


_read_env_file()

from config import settings


def check_env_files() -> bool:
    ok = True
    _print_section("Environment")
    project_root = _project_root()
    env_candidates = [
        project_root / ".env.dev",
        project_root / ".env.prod",
        project_root / ".env",
    ]

    found = [str(p) for p in env_candidates if p.exists()]
    if found:
        _report_ok("env files", ", ".join(os.path.basename(p) for p in found))
    else:
        _report_fail("env files", "No .env file found in project root")
        ok = False

    if settings.APP_ENV:
        _report_ok("APP_ENV", settings.APP_ENV)
    else:
        _report_fail("APP_ENV", "not set")
        ok = False
    return ok


def check_redis() -> bool:
    ok = True
    _print_section("Redis")

    if redis is None:
        _report_fail("redis package", f"Import failed: {REDIS_IMPORT_ERROR}")
        return False

    redis_url = getattr(settings, "REDIS_URL", "")
    if not redis_url:
        _report_fail("REDIS_URL", "not configured")
        return False

    _report_ok("REDIS_URL", redis_url)

    try:
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        result = client.ping()
        if result is True:
            _report_ok("Redis connection", "PING succeeded")
        else:
            _report_fail("Redis connection", "PING returned false")
            ok = False
    except Exception as exc:
        _report_fail("Redis connection", str(exc))
        ok = False
    return ok


def check_qdrant() -> bool:
    ok = True
    _print_section("Qdrant")

    if QdrantClient is None:
        _report_fail("qdrant_client package", f"Import failed: {QDRANT_IMPORT_ERROR}")
        return False

    url = getattr(settings, "QDRANT_URL", "")
    api_key = getattr(settings, "QDRANT_API_KEY", "")
    if not url:
        _report_fail("QDRANT_URL", "not configured")
        return False

    _report_ok("QDRANT_URL", url)

    try:
        client = QdrantClient(url=url, api_key=api_key or None)
        client.get_collections()
        _report_ok("Qdrant connection", "collections API reachable")
    except Exception as exc:
        _report_fail("Qdrant connection", str(exc))
        ok = False
    return ok


def check_embedding_config() -> bool:
    ok = True
    _print_section("Embedding config")
    provider = getattr(settings, "EMBEDDING_PROVIDER", "").lower()
    _report_ok("EMBEDDING_PROVIDER", provider or "not set")

    if provider == "openai":
        key = getattr(settings, "OPENAI_API_KEY", "")
        if not key:
            _report_fail("OPENAI_API_KEY", "missing")
            ok = False
        else:
            _report_ok("OPENAI_API_KEY", "configured")
    elif provider in {"dashscope", "qwen", "ali"}:
        key = getattr(settings, "DASHSCOPE_API_KEY", "")
        if not key:
            _report_fail("DASHSCOPE_API_KEY", "missing")
            ok = False
        else:
            _report_ok("DASHSCOPE_API_KEY", "configured")
    elif provider in {"local", "sentence_transformers", "sentence-transformers"}:
        model = getattr(settings, "LOCAL_EMBEDDING_MODEL", "")
        if not model:
            _report_fail("LOCAL_EMBEDDING_MODEL", "missing")
            ok = False
        else:
            _report_ok("LOCAL_EMBEDDING_MODEL", model)
    else:
        _report_fail("EMBEDDING_PROVIDER", f"unsupported provider: {provider}")
        ok = False

    return ok


def check_rerank_config() -> bool:
    ok = True
    _print_section("Rerank")

    current_model = getattr(settings, "RERANK_MODEL", "")
    is_dashscope = settings.RERANK_MODEL_MAPPING.get(current_model) == settings.RERANK_MODEL_TYPES.DASH_SCOPE

    if not is_dashscope:
        _report_ok("RERANK_MODEL", current_model or "not set")
        return True

    _report_ok("RERANK_MODEL", current_model)

    if TextReRank is None:
        _report_fail("dashscope TextReRank SDK", f"Import failed: {DASHSCOPE_RERANK_IMPORT_ERROR}")
        return False

    api_key = getattr(settings, "DASHSCOPE_API_KEY", "")
    if not api_key:
        _report_fail("DASHSCOPE_API_KEY", "missing")
        return False

    _report_ok("DASHSCOPE_API_KEY", "configured")

    try:
        response = TextReRank.call(
            model=current_model,
            query="test rerank connectivity",
            documents=[
                "This is a positive candidate document for rerank validation.",
                "This is a different unrelated document.",
            ],
            top_n=2,
            api_key=api_key,
            return_documents=True,
        )
        if response.status_code == 200:
            _report_ok("DashScope rerank", "model reachable and returned results")
            return True

        _report_fail("DashScope rerank", f"status_code={response.status_code}, code={getattr(response, 'code', 'unknown')}, message={getattr(response, 'message', 'unknown')}")
        return False
    except Exception as exc:
        _report_fail("DashScope rerank", str(exc))
        return False


def main() -> int:
    checks = []
    checks.append(check_env_files())
    checks.append(check_redis())
    checks.append(check_qdrant())
    checks.append(check_embedding_config())
    checks.append(check_rerank_config())

    failed = sum(1 for ok in checks if not ok)
    _print_section("Summary")
    if failed == 0:
        _report_ok("overall", "All environment checks passed")
        return 0

    _report_fail("overall", f"{failed} check(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
