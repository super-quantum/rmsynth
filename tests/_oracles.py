from __future__ import annotations

from itertools import combinations

from rmsynth_reference import CNOT, Circuit


def run_circuit(circuit: Circuit, state: int) -> tuple[int, int]:
    phase = 0
    for operation in circuit.operations:
        if isinstance(operation, CNOT):
            control = (state >> operation.control) & 1
            state ^= control << operation.target
        elif (state >> operation.qubit) & 1:
            phase = (phase + operation.exponent) & 7
    return state, phase


def evaluate_coefficients(coefficients: tuple[int, ...], state: int) -> int:
    total = 0
    for index, coefficient in enumerate(coefficients):
        parity_mask = index + 1
        total += coefficient * ((parity_mask & state).bit_count() & 1)
    return total & 7


def rank(rows: tuple[int, ...], columns: int) -> int:
    work = list(rows)
    pivot_row = 0
    for column in range(columns):
        for candidate in range(pivot_row, len(work)):
            if work[candidate] & (1 << column):
                break
        else:
            continue
        work[pivot_row], work[candidate] = work[candidate], work[pivot_row]
        for row in range(pivot_row + 1, len(work)):
            if work[row] & (1 << column):
                work[row] ^= work[pivot_row]
        pivot_row += 1
    return pivot_row


def rm_terms(qubits: int, order: int) -> tuple[int, ...]:
    if order < 0:
        return ()
    terms = [0]
    for degree in range(1, order + 1):
        terms.extend(sum(1 << bit for bit in bits) for bits in combinations(range(qubits), degree))
    return tuple(sorted(terms))


def rm_rows(qubits: int, order: int) -> tuple[int, ...]:
    result = []
    for term in rm_terms(qubits, order):
        values = 0
        for parity_mask in range(1, 1 << qubits):
            if term == 0 or parity_mask & term == term:
                values |= 1 << (parity_mask - 1)
        result.append(values)
    return tuple(result)


def nearest_codeword(received: int, rows: tuple[int, ...]) -> tuple[int, int, int]:
    words = {0}
    for row in rows:
        words |= {word ^ row for word in tuple(words)}
    distances = {word: (received ^ word).bit_count() for word in words}
    distance = min(distances.values())
    nearest = min(
        word for word, candidate_distance in distances.items() if candidate_distance == distance
    )
    ties = sum(candidate_distance == distance for candidate_distance in distances.values())
    return nearest, distance, ties
