from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ._version import __version__
from .limits import (
    MAX_DECODER_CANDIDATES,
    MAX_INPUT_BYTES,
    MAX_OPERATIONS,
    MAX_OPTIMIZER_QUBITS,
    MAX_VERIFIER_QUBITS,
)


@dataclass(frozen=True, slots=True)
class OptimizationReport:
    """Stable metadata for one optimization attempt."""

    status: str
    reason: str
    qubits: int
    reed_muller_order: int
    before_t_count: int
    after_t_count: int
    candidates: int
    ties: int
    input_digest: str
    output_digest: str
    semantic_digest: str
    verified: bool
    schema: str = "rmsynth-reference/report-v1"
    tool_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["limits"] = {
            "decoder_candidates": MAX_DECODER_CANDIDATES,
            "input_bytes": MAX_INPUT_BYTES,
            "operations": MAX_OPERATIONS,
            "optimizer_qubits": MAX_OPTIMIZER_QUBITS,
            "verifier_qubits": MAX_VERIFIER_QUBITS,
        }
        return result
