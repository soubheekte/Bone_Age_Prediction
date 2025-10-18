import logging
import logging.handlers
import os
import json
import time
import datetime
from typing import Optional
from config.config import Config
import inspect


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


def _cleanup_old_logs(log_dir: str, base_name: str, days: int = 7) -> None:
    """
    Remove files named like base_name or base_name.* older than `days`.
    This is an extra safety cleanup in addition to TimedRotatingFileHandler's backupCount.
    """
    cutoff = time.time() - days * 86400
    try:
        for fname in os.listdir(log_dir):
            # only consider files that are exactly base or start with base + "."
            if not (fname == base_name or fname.startswith(base_name + ".")):
                continue
            fpath = os.path.join(log_dir, fname)
            try:
                if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)
            except Exception:
                # ignore individual file errors
                pass
    except Exception:
        # ignore directory listing errors
        pass


# new: handler that names active file with current hour suffix and swaps files when hour changes
class HourlyFileHandler(logging.Handler):
    """
    Active log filename includes current hour, e.g. pipeline_20251018_14.log.
    On hour change the handler swaps to a new FileHandler and runs cleanup.
    """

    def __init__(
        self,
        base_log_file: str,
        encoding: str = "utf-8",
        retention_days: int = 7,
        level: int = logging.NOTSET,
    ):
        super().__init__(level)
        self.base_log_file = base_log_file
        self.encoding = encoding
        self.retention_days = retention_days
        self._current_hour = None
        self._file_handler: logging.FileHandler | None = None
        self._ensure_handler()

    def _current_filename(self) -> str:
        # insert _YYYYmmdd_HH before extension
        dirname = os.path.dirname(self.base_log_file) or "."
        base = os.path.basename(self.base_log_file)
        name, ext = os.path.splitext(base)
        now = datetime.datetime.now()
        suffix = now.strftime("%Y%m%d_%H")
        return os.path.join(dirname, f"{name}_{suffix}{ext}")

    def _ensure_handler(self):
        filename = self._current_filename()
        hour_marker = filename  # unique per hour
        if hour_marker != self._current_hour:
            # swap handlers
            if self._file_handler:
                try:
                    self._file_handler.close()
                except Exception:
                    pass
            # ensure directory exists
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            self._file_handler = logging.FileHandler(filename, encoding=self.encoding)
            self._file_handler.setLevel(self.level)
            # copy our formatter if already set
            if self.formatter:
                self._file_handler.setFormatter(self.formatter)
            self._current_hour = hour_marker
            # run cleanup of old files for base name
            try:
                base_name = os.path.basename(self.base_log_file)
                _cleanup_old_logs(
                    os.path.dirname(filename) or ".", base_name, days=self.retention_days
                )
            except Exception:
                pass

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # ensure right file for current hour
            self._ensure_handler()
            if self._file_handler:
                self._file_handler.emit(record)
        except Exception:
            self.handleError(record)

    def setFormatter(self, fmt: logging.Formatter) -> None:
        super().setFormatter(fmt)
        # propagate to internal handler immediately if exists
        if getattr(self, "_file_handler", None):
            try:
                self._file_handler.setFormatter(fmt)
            except Exception:
                pass

    def close(self) -> None:
        try:
            if self._file_handler:
                try:
                    self._file_handler.close()
                finally:
                    self._file_handler = None
        finally:
            super().close()


def _detect_caller() -> tuple[str, str]:
    """
    Inspect the stack to find the first frame outside this module and return
    (logger_name, default_log_file).
    logger_name -> module[.function] (function omitted if top-level)
    default_log_file -> logs/<module>.log
    """
    this_file = os.path.abspath(__file__)
    stack = inspect.stack()
    for frame_info in stack[2:]:
        fn = os.path.abspath(frame_info.filename)
        if fn == this_file:
            continue
        module_name = os.path.splitext(os.path.basename(fn))[0]
        func = frame_info.function
        if func and func != "<module>":
            logger_name = f"{module_name}.{func}"
        else:
            logger_name = module_name
        default_log_file = os.path.join("logs", f"{module_name}.log")
        return logger_name, default_log_file
    # fallback
    return ("root", os.path.join("logs", "pipeline.log"))


def get_logger(
    name: Optional[str] = None,
    log_file: Optional[str] = None,
    level: Optional[int] = None,
    rotate_mb: int = 5,
    backup_count: int = 168,
    use_json: bool = False,
) -> logging.Logger:
    """
    Returns a configured logger.
    - If name or log_file is not provided, auto-detect from the caller.
    - Creates `logs/` directory if missing.
    - Adds an HourlyFileHandler (hourly file name suffix) and a StreamHandler.
    - Keeps about one week's worth of hourly logs (backup_count defaults to 7*24).
    - Honors Config.DEBUG from [`config.Config`](config/config.py).
    """
    # detect caller info if not provided
    if name is None or log_file is None:
        detected_name, detected_log_file = _detect_caller()
        if name is None:
            name = detected_name
        if log_file is None:
            log_file = detected_log_file

    cfg = Config()
    resolved_level = (
        level
        if level is not None
        else (logging.DEBUG if cfg.DEBUG else logging.INFO)
    )

    # ensure directory exists
    log_dir = os.path.dirname(log_file) or "logs"
    os.makedirs(log_dir, exist_ok=True)

    # cleanup any old log files older than 7 days (safety)
    base_name = os.path.basename(log_file)
    try:
        _cleanup_old_logs(log_dir, base_name, days=7)
    except Exception:
        pass

    logger = logging.getLogger(name)
    logger.setLevel(resolved_level)
    logger.propagate = False

    abs_base_log_file = os.path.abspath(log_file)

    if not any(
        isinstance(h, HourlyFileHandler)
        and getattr(h, "base_log_file", "") == abs_base_log_file
        for h in logger.handlers
    ):
        formatter = _make_formatter(use_json)

        # use HourlyFileHandler so active filename includes hour suffix
        fh = HourlyFileHandler(
            base_log_file=log_file,
            encoding="utf-8",
            retention_days=7,
            level=resolved_level,
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(resolved_level)
        # console output as readable text
        sh.setFormatter(_make_formatter(False))
        logger.addHandler(sh)

    try:
        import tensorflow as tf  # optional integration if TF is available
    except Exception:
        tf = None

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

