"""
Application-wide logging setup.

Logs to both console and a rotating file (logs/app.log), with a consistent
format that includes timestamp, logger name, level, and message. Third-party
libraries (httpx, chromadb, etc.) are turned down to WARNING to keep signal
high.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

from app.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    os.makedirs("logs", exist_ok=True)

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    formatter = logging.Formatter(log_format)

    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    # Avoid duplicate handlers on reload
    if root.handlers:
        return

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        "logs/app.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for noisy_logger in ("httpx", "httpcore", "chromadb", "google", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
