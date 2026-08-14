from __future__ import annotations

from dataclasses import dataclass

from . import _native
from .errors import ResourceLimitError, ValidationError
from .reed_muller import _validate_parameter_types


@dataclass(frozen=True, slots=True)
class DecodeResult:
    codeword: int
    selected_terms: tuple[int, ...]
    distance: int
    candidates: int
    ties: int


def decode_exact(received: int, qubits: int, order: int) -> DecodeResult:
    if type(received) is not int or received < 0:
        raise ValidationError("received word must be a nonnegative integer")
    _validate_parameter_types(qubits, order)
    try:
        result = _native.decode_exact(received, qubits, order)
    except _native.ValidationError as error:
        raise ValidationError(str(error)) from error
    except _native.ResourceLimitError as error:
        raise ResourceLimitError(str(error)) from error
    return DecodeResult(
        result.codeword,
        tuple(result.selected_terms),
        result.distance,
        result.candidates,
        result.ties,
    )
