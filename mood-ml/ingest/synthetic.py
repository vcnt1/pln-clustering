"""Deprecated: the labeled dataset arrives ready (contract dc-1), so this generator is not used.

Kept only as a reference for data-model §4.1. See specs/02-ingest-validate.md.
"""


def generate_synthetic_conversations(n_conversations: int) -> list[dict]:
    raise NotImplementedError("deprecated: use ingest.validate on a dc-1 .jsonl file")
