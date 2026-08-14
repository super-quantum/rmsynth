from __future__ import annotations

import pytest

from rmsynth_reference.decoder import decode_exact
from rmsynth_reference.errors import ResourceLimitError, ValidationError
from rmsynth_reference.model import PhasePolynomial
from rmsynth_reference.reed_muller import rm_basis_terms, rm_dimension, rm_generator_rows
from rmsynth_reference.verify import verify_phase_polynomials
from tests._oracles import nearest_codeword, rm_rows


@pytest.mark.parametrize(
    ("qubits", "order", "dimension"),
    [(4, -1, 0), (4, 0, 1), (5, 1, 6), (5, 2, 16)],
)
def test_known_dimensions(qubits: int, order: int, dimension: int) -> None:
    assert rm_dimension(qubits, order) == dimension
    assert len(rm_basis_terms(qubits, order)) == dimension


@pytest.mark.parametrize(("qubits", "order"), [(3, -1), (4, 0), (5, 1)])
def test_generator_rows_match_oracle(qubits: int, order: int) -> None:
    assert rm_generator_rows(qubits, order) == rm_rows(qubits, order)


def test_four_qubit_decoder_exhaustively() -> None:
    rows = rm_rows(4, 0)
    for received in range(1 << 15):
        expected_word, expected_distance, expected_ties = nearest_codeword(received, rows)
        result = decode_exact(received, 4, 0)
        assert (result.codeword, result.distance, result.ties) == (
            expected_word,
            expected_distance,
            expected_ties,
        )


def test_five_qubit_decoder_corpus() -> None:
    samples = [0, (1 << 31) - 1, *(1 << bit for bit in range(31)), 33023]
    samples.extend((index * 0x45D9F3B) & ((1 << 31) - 1) for index in range(1, 65))
    rows = rm_rows(5, 1)
    for received in samples:
        expected_word, expected_distance, expected_ties = nearest_codeword(received, rows)
        result = decode_exact(received, 5, 1)
        assert (result.codeword, result.distance, result.ties) == (
            expected_word,
            expected_distance,
            expected_ties,
        )
        assert (received ^ result.codeword).bit_count() == result.distance


def test_tie_breaking_is_stable() -> None:
    result = decode_exact(33023, 5, 1)
    assert result.codeword == 32767
    assert result.distance == 8
    assert result.ties == 2
    assert result == decode_exact(33023, 5, 1)


def test_rm_codewords_are_zero_phase() -> None:
    rows = rm_generator_rows(5, 1)
    zero = PhasePolynomial(5, (0,) * 31)
    for selection in range(64):
        codeword = 0
        for index, row in enumerate(rows):
            if selection >> index & 1:
                codeword ^= row
        correction = PhasePolynomial(5, tuple(codeword >> index & 1 for index in range(31)))
        assert verify_phase_polynomials(zero, correction).equivalent


def test_candidate_limit_is_checked() -> None:
    with pytest.raises(ResourceLimitError, match="2147483648"):
        decode_exact(0, 5, 4)


@pytest.mark.parametrize("received", [-1, True, 1 << 15])
def test_received_word_validation(received: object) -> None:
    with pytest.raises(ValidationError):
        decode_exact(received, 4, 0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("qubits", "order"),
    [(0, 0), (2, -2), (2, 2), (2, 3), (True, 0), (2, False), (2, 0.0)],
)
def test_parameter_validation(qubits: object, order: object) -> None:
    with pytest.raises(ValidationError):
        rm_dimension(qubits, order)  # type: ignore[arg-type]


def test_decoder_parameter_type_validation() -> None:
    with pytest.raises(ValidationError, match="qubits must be an integer"):
        decode_exact(0, True, 0)


def test_decoder_rejects_dimension_before_bit_shifts() -> None:
    with pytest.raises(ValidationError, match="between 1 and 5"):
        decode_exact(0, 1_000_000, 0)
