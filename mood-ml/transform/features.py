"""T3: builds the model input blocks from InferRequest.history (see specs/04-transform.md)."""

from transform.clean import clean_message
from transform.mask import mask_pii

FEATURE_SPEC_VERSION = "fs-1"
MAX_HISTORY_SIZE = 30


def extract_features(history: list[dict]) -> dict:
    if not history:
        raise ValueError("history must not be empty")
    if len(history) > MAX_HISTORY_SIZE:
        raise ValueError(f"history must have at most {MAX_HISTORY_SIZE} items, got {len(history)}")
    for item in history:
        if item["role"] != "customer":
            raise ValueError(f"history item has role={item['role']!r}, expected 'customer'")

    cleaned = [mask_pii(clean_message(item["text"])) for item in history]

    return {"text_clean": cleaned[-1], "context_clean": cleaned[:-1]}
