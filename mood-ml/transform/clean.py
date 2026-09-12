"""Cleans raw messages and extracts features for the mood model."""


def clean_message(text: str) -> str:
    # TODO: define cleaning rules (boilerplate removal, PII masking, etc.).
    return text.strip()


def extract_features(messages: list[str]) -> list[float]:
    # TODO: replace with the chosen feature/embedding strategy.
    raise NotImplementedError
