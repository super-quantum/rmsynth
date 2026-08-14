# Scope and limits

Version 0.1 has one optimizer: exact T-count reduction for small CNOT-phase circuits.

## Supported input

- 1 to 5 qubits for optimization
- 1 to 10 qubits for inspection and equivalence checking
- CNOT gates
- diagonal gates `diag(1, exp(iπk/4))` with `0 <= k <= 7`
- at most 10,000 operations and 1 MiB of JSON input

Optimization preserves the phase function and the final invertible GF(2) linear map. A result is
returned only if its T-count is lower than that of the input circuit. Otherwise the input circuit is
returned unchanged.

## Not supported

- T-depth optimization or scheduling
- native algorithms other than exact decoding and GF(2) linear-map synthesis
- arbitrary Clifford or nonlinear gates
- heuristic decoder portfolios or autotuning
- optimization above five qubits
- claims about production-scale performance

A request beyond a fixed limit fails before expensive enumeration. There is no automatic switch to
an undocumented algorithm.

This repository is maintained as a reference implementation. It has a separate release cycle and
API from other RMSynth software.
