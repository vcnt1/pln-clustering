"""T3: builds the model input blocks from InferRequest.history (see specs/04-transform.md)."""

from typing import Any

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


def join_context_list(column: Any) -> list[str]:
    """Joins each row's context_clean list into a single string (space
    separator, "" if empty). Lives inside the serialized train.train Pipeline
    (TN-R09), so train and infer share the exact same behavior automatically.

    Deliberately kept here rather than in train/train.py: joblib pickles a
    plain function by (module, qualname) reference, and train/train.py is
    also a CLI entry point (`python -m train.train` runs it with
    __name__ == "__main__"). A function defined there would get pickled
    against whichever module happened to be "__main__" at train time, which
    breaks as soon as it's unpickled from a *different* process's __main__
    (e.g. `python -m evaluate.metrics` or `python -m registry.registry`).
    transform/features.py is a plain library module, never a -m entry point,
    so its functions always pickle against the stable "transform.features"
    reference.

    Uses `len(item) == 0` rather than `if item`: a Parquet round-trip through
    pyarrow turns the list column into numpy arrays, not plain Python lists,
    and truth-testing a multi-element numpy array raises ValueError."""
    return ["" if item is None or len(item) == 0 else " ".join(str(x) for x in item) for item in column]
