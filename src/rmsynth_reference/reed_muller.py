from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from . import _native
from .errors import ValidationError

T = TypeVar("T")


def rm_dimension(qubits: int, order: int) -> int:
    return int(_call(_native.rm_dimension, qubits, order))


def rm_basis_terms(qubits: int, order: int) -> tuple[int, ...]:
    result = _call(_native.rm_basis_terms, qubits, order)
    return tuple(result)


def rm_generator_rows(qubits: int, order: int) -> tuple[int, ...]:
    result = _call(_native.rm_generator_rows, qubits, order)
    return tuple(result)


def _call(function: Callable[[int, int], T], qubits: int, order: int) -> T:
    _validate_parameter_types(qubits, order)
    try:
        return function(qubits, order)
    except _native.ValidationError as error:
        raise ValidationError(str(error)) from error


def _validate_parameter_types(qubits: int, order: int) -> None:
    if type(qubits) is not int:
        raise ValidationError("qubits must be an integer")
    if type(order) is not int:
        raise ValidationError("order must be an integer")
