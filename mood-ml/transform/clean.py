"""T1: normalizes raw message text (see specs/04-transform.md)."""

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CATEGORIES = {"Cc", "Cf"}


def clean_message(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped

    collapsed = _WHITESPACE_RE.sub(" ", stripped)
    normalized = unicodedata.normalize("NFC", collapsed)
    without_control = "".join(
        ch for ch in normalized if unicodedata.category(ch) not in _CONTROL_CATEGORIES
    )

    if not without_control.strip():
        return stripped

    return without_control
