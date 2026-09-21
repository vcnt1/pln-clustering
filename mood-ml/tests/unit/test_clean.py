import pytest

from transform.clean import clean_message

ZERO_WIDTH_SPACE = "\u200b"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("  Oi   tudo bem?  ", "Oi tudo bem?"),
        ("ABSURDO!!!", "ABSURDO!!!"),
        ("Amei\ttudo\n\ndemais!!", "Amei tudo demais!!"),
        ("Otimo \U0001f600 valeu!", "Otimo \U0001f600 valeu!"),
        ("???", "???"),
    ],
)
def test_clean_message_cases(text: str, expected: str) -> None:
    assert clean_message(text) == expected


def test_clean_message_never_returns_empty_for_control_only_text() -> None:
    text = ZERO_WIDTH_SPACE * 2  # survives str.strip(), but not the T1 control-char pass
    result = clean_message(text)
    assert result == text.strip()
    assert result != ""


def test_clean_message_applies_nfc_normalization() -> None:
    decomposed = "e" + "́"  # "e" + combining acute accent
    composed = "é"  # precomposed "e" with acute accent
    assert clean_message(decomposed) == composed


def test_clean_message_removes_control_characters() -> None:
    text = f"Ola{ZERO_WIDTH_SPACE}mundo"
    assert ZERO_WIDTH_SPACE not in clean_message(text)


def test_clean_message_is_pure() -> None:
    text = "  mesmo texto  "
    assert clean_message(text) == clean_message(text)
