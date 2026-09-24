"""Base exception shared by every pipeline layer (see specs/10-common.md)."""


class PipelineError(Exception):
    """`code` is the layer's spec-mandated error code; each layer subclasses
    this so its CLI can map the exception type to an exit code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
