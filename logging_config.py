import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
import datetime


class DualTimestampFormatter(logging.Formatter):
    """Formatter with ISO UTC and local timestamps."""

    def format(self, record):
        utcnow = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
        iso_utc = utcnow.isoformat(timespec="microseconds").replace("+00:00", "Z")
        asctime = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        levelname = record.levelname
        message = record.getMessage()
        return f"{iso_utc} {asctime} - {levelname} - {message}"


def setup_logging():
    """Configure root logger with file rotation and stdout."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    formatter = DualTimestampFormatter()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    file_handler = RotatingFileHandler(
        logs_dir / "app.log", maxBytes=10_000_000, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

