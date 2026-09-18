"""T3: builds the model input blocks from InferRequest.history (see specs/04-transform.md)."""


def extract_features(history: list[dict]) -> dict:
    # TODO: apply mask_pii(clean_message(text)) to each item; return text_clean: str (last item)
    # and context_clean: list[str] (previous items, [] if none) (ADR-0008).
    raise NotImplementedError
