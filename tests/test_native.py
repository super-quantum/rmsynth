from __future__ import annotations

import pytest

from rmsynth_reference import _native


def test_native_reed_muller_surface() -> None:
    assert _native.rm_dimension(5, 1) == 6
    assert _native.rm_basis_terms(5, 1) == [0, 1, 2, 4, 8, 16]
    assert _native.rm_generator_rows(4, 0) == [0x7FFF]


def test_native_decoder_result() -> None:
    result = _native.decode_exact(33023, 5, 1)
    assert result.codeword == 32767
    assert result.selected_terms == [0, 16]
    assert result.distance == 8
    assert result.candidates == 64
    assert result.ties == 2


def test_native_error_boundary() -> None:
    with pytest.raises(_native.ValidationError, match="between 1 and 5"):
        _native.rm_dimension(0, 0)
    with pytest.raises(_native.ValidationError, match="code length"):
        _native.decode_exact(1 << 15, 4, 0)
    with pytest.raises(_native.ResourceLimitError, match="2147483648"):
        _native.decode_exact(0, 5, 4)


def test_native_linear_map_synthesis() -> None:
    assert _native.synthesize_linear_map([1, 2, 4]) == []
    gates = _native.synthesize_linear_map([2, 1])
    rows = [1, 2]
    for control, target in gates:
        rows[target] ^= rows[control]
    assert rows == [2, 1]
