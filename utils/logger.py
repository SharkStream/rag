from __future__ import annotations

import logging

from config import settings


def configure_logging() -> logging.Logger:
    level = logging.DEBUG if settings.DEBUG else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d]: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger = logging.getLogger("rag_app")
    logger.setLevel(level)
    return logger

logger = configure_logging()