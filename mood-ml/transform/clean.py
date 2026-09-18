"""T1: normalizes raw message text (see specs/04-transform.md)."""


def clean_message(text: str) -> str:
    # TODO: define cleaning rules in spec 04 (whitespace/newlines; keep case, punctuation, emojis).
    return text.strip()
