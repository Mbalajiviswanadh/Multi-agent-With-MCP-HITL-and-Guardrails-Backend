"""
src/core/logging_config.py
--------------------------
Centralised logger factory. Every module in the project calls
`get_logger(__name__)` and gets a consistently formatted logger that
writes:
  - Clean, color-coded, easy-to-scan output to the console (stdout)
  - Plain, structured records to `agent_runs.log` (no ANSI escape codes)

Also silences noisy third-party library logs (httpx, httpcore, qdrant_client,
posthog, urllib3) so testing output stays focused on agent execution.
"""
from __future__ import annotations

import logging
import os
import re
import sys

# Enable ANSI colors on Windows 10/11 terminals
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_MAGENTA = "\033[95m"
_CYAN = "\033[96m"
_WHITE = "\033[97m"

_TAG_COLORS = {
    "SUPERVISOR": _CYAN,
    "INPUT_GUARDRAIL": _CYAN,
    "OUTPUT_GUARDRAIL": _CYAN,
    "GUARDRAIL": _CYAN,
    "USER_MCP_AGENT": _MAGENTA,
    "MCP_API": _MAGENTA,
    "MCP_OAUTH": _MAGENTA,
    "MCP_CLIENT": _MAGENTA,
    "FLIGHT_AGENT": _BLUE,
    "HOTEL_AGENT": _BLUE,
    "WEATHER_AGENT": _BLUE,
    "BUDGET_AGENT": _BLUE,
    "ITINERARY_AGENT": _BLUE,
    "HUMAN_APPROVAL": _YELLOW,
    "FINAL_AGENT": _GREEN,
    "CHAT_AGENT": _BLUE,
    "MEM0": _GREEN,
    "VECTOR": _GREEN,
    "PROFILE": _GREEN,
    "SUMMARY": _GREEN,
    "LLM": _YELLOW,
    "STREAM START": _BOLD + _WHITE,
    "STREAM FINISHED": _BOLD + _WHITE,
    "COMPLETE": _BOLD + _GREEN,
    "RESUME": _BOLD + _WHITE,
    "FAILED": _BOLD + _RED,
}

_TAG_RE = re.compile(r"^\[([A-Z0-9_ ]+)\]")
_NOISE_CONFIGURED = False


def _silence_noisy_loggers() -> None:
    """Reduce chatter from HTTP/vector/telemetry libraries during testing."""
    global _NOISE_CONFIGURED
    if _NOISE_CONFIGURED:
        return
    _NOISE_CONFIGURED = True

    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "qdrant_client",
        "posthog",
        "backoff",
        "langsmith",
        "fastembed",
        "onnxruntime",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class _PrettyConsoleFormatter(logging.Formatter):
    """Formats console logs with subtle timestamps, colored component tags, and clear error badges."""

    def __init__(self, use_color: bool = True):
        super().__init__(datefmt="%H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, self.datefmt)
        msg = record.getMessage()

        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"

        if not self.use_color:
            prefix = f"[{ts}]"
            if record.levelno >= logging.ERROR:
                return f"{prefix} ERROR | {msg}"
            if record.levelno >= logging.WARNING:
                return f"{prefix} WARN  | {msg}"
            return f"{prefix} {msg}"

        ts_part = f"{_DIM}[{ts}]{_RESET}"

        # Level badge for warnings/errors so problems pop out immediately
        if record.levelno >= logging.ERROR:
            lvl_part = f" {_BOLD}{_RED}ERR{_RESET} "
        elif record.levelno >= logging.WARNING:
            lvl_part = f" {_YELLOW}WRN{_RESET} "
        else:
            lvl_part = " "

        # Colorize leading [TAG] if present
        m = _TAG_RE.match(msg)
        if m:
            tag = m.group(1)
            color = _TAG_COLORS.get(tag, _CYAN)
            rest = msg[m.end():]
            # Subtle highlight for OK / error statuses in the message body
            rest = rest.replace(" -> success", f" -> {_GREEN}success{_RESET}")
            rest = rest.replace(" -> error", f" -> {_RED}error{_RESET}")
            if " OK " in rest:
                rest = rest.replace(" OK ", f" {_GREEN}OK{_RESET} ", 1)
            if " FAIL " in rest:
                rest = rest.replace(" FAIL ", f" {_RED}FAIL{_RESET} ", 1)
            msg = f"{color}{_BOLD}[{tag}]{_RESET}{rest}"
        elif msg.startswith("===") or msg.startswith("---"):
            msg = f"{_DIM}{msg}{_RESET}"

        return f"{ts_part}{lvl_part}{msg}"


def get_logger(name: str = "TripMateAI") -> logging.Logger:
    """
    Return a logger for *name*. Handlers are added only once; subsequent
    calls with the same name reuse the existing handlers.
    """
    _silence_noisy_loggers()

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent duplicate lines via root logger

    use_color = sys.stdout.isatty() or os.getenv("FORCE_COLOR", "1") == "1"
    console_fmt = _PrettyConsoleFormatter(use_color=use_color)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    # File handler (plain text, no ANSI codes)
    log_file = os.getenv("LOG_FILE", "agent_runs.log")
    try:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(file_fmt)
        logger.addHandler(fh)
    except OSError as exc:
        logger.warning("Could not open log file %s: %s", log_file, exc)

    return logger
