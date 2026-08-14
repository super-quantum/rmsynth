from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .errors import ResourceLimitError, ValidationError
from .limits import MAX_OPERATIONS, MAX_VERIFIER_QUBITS


def _integer(name: str, value: object, minimum: int, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise ValidationError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = (
            f" between {minimum} and {maximum}" if maximum is not None else f" at least {minimum}"
        )
        raise ValidationError(f"{name} must be{bound}")
    return value


@dataclass(frozen=True, slots=True)
class CNOT:
    """A controlled-NOT gate."""

    control: int
    target: int

    def __post_init__(self) -> None:
        _integer("control", self.control, 0)
        _integer("target", self.target, 0)
        if self.control == self.target:
            raise ValidationError("CNOT control and target must differ")


@dataclass(frozen=True, slots=True)
class Phase:
    """A diagonal phase gate with exponent measured in multiples of pi/4."""

    qubit: int
    exponent: int

    def __post_init__(self) -> None:
        _integer("qubit", self.qubit, 0)
        _integer("phase exponent", self.exponent, 0, 7)


Operation: TypeAlias = CNOT | Phase


@dataclass(frozen=True, slots=True)
class Circuit:
    """An immutable CNOT-phase circuit."""

    qubits: int
    operations: tuple[Operation, ...] = ()

    def __post_init__(self) -> None:
        _integer("qubits", self.qubits, 1, MAX_VERIFIER_QUBITS)
        if not isinstance(self.operations, tuple):
            raise ValidationError("operations must be a tuple")
        if len(self.operations) > MAX_OPERATIONS:
            raise ResourceLimitError(f"circuits may contain at most {MAX_OPERATIONS} operations")
        for operation in self.operations:
            if not isinstance(operation, (CNOT, Phase)):
                raise ValidationError("operations must contain only CNOT and Phase gates")
            indices = (
                (operation.control, operation.target)
                if isinstance(operation, CNOT)
                else (operation.qubit,)
            )
            if any(index >= self.qubits for index in indices):
                raise ValidationError("gate qubit index is outside the circuit")

    @property
    def t_count(self) -> int:
        return sum(isinstance(op, Phase) and op.exponent % 2 == 1 for op in self.operations)


@dataclass(frozen=True, slots=True)
class PhasePolynomial:
    """Z8 coefficients indexed by nonzero parity mask minus one."""

    qubits: int
    coefficients: tuple[int, ...]

    def __post_init__(self) -> None:
        _integer("qubits", self.qubits, 1, MAX_VERIFIER_QUBITS)
        if not isinstance(self.coefficients, tuple):
            raise ValidationError("coefficients must be a tuple")
        expected = (1 << self.qubits) - 1
        if len(self.coefficients) != expected:
            raise ValidationError(f"expected {expected} phase coefficients")
        for coefficient in self.coefficients:
            _integer("phase coefficient", coefficient, 0, 7)

    @property
    def t_count(self) -> int:
        return sum(coefficient % 2 for coefficient in self.coefficients)


def _rank(rows: tuple[int, ...], columns: int) -> int:
    work = list(rows)
    rank = 0
    for column in range(columns):
        pivot = next((row for row in range(rank, len(work)) if work[row] >> column & 1), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        for row in range(len(work)):
            if row != rank and work[row] >> column & 1:
                work[row] ^= work[rank]
        rank += 1
    return rank


@dataclass(frozen=True, slots=True)
class LinearMap:
    """An invertible GF(2) map whose rows define output parities."""

    rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple) or not self.rows:
            raise ValidationError("linear-map rows must be a nonempty tuple")
        qubits = len(self.rows)
        if qubits > MAX_VERIFIER_QUBITS:
            raise ValidationError(f"linear maps support at most {MAX_VERIFIER_QUBITS} qubits")
        for row in self.rows:
            _integer("linear-map row", row, 0, (1 << qubits) - 1)
        if _rank(self.rows, qubits) != qubits:
            raise ValidationError("linear map must be invertible over GF(2)")

    @classmethod
    def identity(cls, qubits: int) -> LinearMap:
        _integer("qubits", qubits, 1, MAX_VERIFIER_QUBITS)
        return cls(tuple(1 << index for index in range(qubits)))

    @property
    def qubits(self) -> int:
        return len(self.rows)


@dataclass(frozen=True, slots=True)
class LinearPhaseProgram:
    """A phase polynomial paired with its terminal linear map."""

    phase_polynomial: PhasePolynomial
    linear_map: LinearMap

    def __post_init__(self) -> None:
        if self.phase_polynomial.qubits != self.linear_map.qubits:
            raise ValidationError("phase polynomial and linear map must have equal dimensions")

    @property
    def qubits(self) -> int:
        return self.phase_polynomial.qubits
