"""T2: masks PII (CPF, phone, e-mail) in message text (see specs/04-transform.md).

Only the detection/masking surface of T2 is implemented here. T1 (clean_message)
and T3 (extract_features) remain stubs for a later phase.
"""

import re

_EMAIL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CPF_RE = re.compile(r"\d{3}\.\d{3}\.\d{3}-\d{2}")
_PHONE_MARKED_RE = re.compile(r"(?:\+55[\s-]?)?(?:\(?\d{2}\)?[\s-]?)?9?\d{4}-\d{4}")
_PHONE_RAW_RE = re.compile(r"\b\d{10,11}\b")

# Applied in this order, each pass over the output of the previous one
# (04-transform.md §3.4): a marker inserted by an earlier pass is never a
# digit/"@" sequence, so it can't be recaptured by a later pattern.
_PASSES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("email", _EMAIL_RE, "<EMAIL>"),
    ("cpf", _CPF_RE, "<CPF>"),
    ("phone", _PHONE_MARKED_RE, "<TEL>"),
    ("phone", _PHONE_RAW_RE, "<TEL>"),
)


def mask_pii(text: str) -> str:
    """Replaces CPF/phone/e-mail occurrences with fixed markers. Idempotent."""
    for _kind, pattern, marker in _PASSES:
        text = pattern.sub(marker, text)
    return text


def count_pii(text: str) -> dict[str, int]:
    """Counts CPF/phone/e-mail occurrences using the same patterns as mask_pii."""
    counts = {"cpf": 0, "phone": 0, "email": 0}
    for kind, pattern, marker in _PASSES:
        matches = pattern.findall(text)
        counts[kind] += len(matches)
        text = pattern.sub(marker, text)
    return counts
