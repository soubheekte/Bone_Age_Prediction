import logging
import logging.handlers
import os
import json
from typing import Optional
from config.config import Config

try:
    import tensorflow as tf  # optional integration if TF is available
except Exception:
    tf = None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "funcName": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _make_formatter(use_json: bool) -> logging.Formatter:
    if use_json:
        return JsonFormatter()
    fmt = (
        "%(asctime)s %(levelname)s [%(name)s] "
        "%(module)s:%(lineno)d - %(message)s"
    )
    return logging.Formatter(fmt)


def get_logger(
    name: str,
    log_file: str = "logs/pipeline.log",
    level: Optional[int] = None,
    rotate_mb: int = 5,
    backup_count: int = 5,
    use_json: bool = False,
) -> logging.Logger:
    """
    Returns a configured logger.
    - Creates `logs/` directory if missing.
    - Adds a RotatingFileHandler and a StreamHandler.
    - Honors Config.DEBUG from [`config.Config`](config/config.py).
    """
    cfg = Config()
    resolved_level = (
        level
        if level is not None
        else (logging.DEBUG if cfg.DEBUG else logging.INFO)
    )

    # ensure directory exists
    log_dir = os.path.dirname(log_file) or "logs"
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.propagate = False

    if not any(
        isinstance(h, logging.handlers.RotatingFileHandler)
        and getattr(h, "baseFilename", "") == os.path.abspath(log_file)
        for h in logger.handlers
    ):
        formatter = _make_formatter(use_json)

        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=rotate_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(resolved_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(resolved_level)
        # console output as readable text
        sh.setFormatter(_make_formatter(False))
        logger.addHandler(sh)

    # integrate with TensorFlow logger if available
    if tf is not None:
        try:
            tfl = tf.get_logger()
            tfl.setLevel(resolved_level)
        except Exception:
            # ignore TF logger configuration errors
            pass

    return logger


def log_exception(
    logger: logging.Logger,
    exc: Exception,
    msg: str = "Unhandled exception",
) -> None:
    # pass a proper exc_info tuple so the traceback is preserved
    logger.exception(msg, exc_info=(type(exc), exc, exc.__traceback__))

    return None

