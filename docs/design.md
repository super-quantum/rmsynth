# Design and correctness

For `n` input bits, a phase polynomial stores one coefficient in `Z8` for each nonzero parity mask.
A circuit is represented by that polynomial and by an invertible binary matrix describing its final
linear action.

The implementation has one native module. Its public C++ library contains punctured Reed–Muller
basis generation, bounded exact decoding, and GF(2) linear-map synthesis. The Python layer owns the
circuit model, phase-gadget synthesis, serialization, reporting, and verification.

## Extraction

Extraction starts with one input mask on each wire. A CNOT XORs the control mask into the target
mask. A phase gate adds its exponent to the coefficient for the current mask on its wire. The wire
masks at the end are the rows of the final linear map.

## Optimization

The odd coefficients form a received word. The C++ core enumerates the punctured
`RM(n - 4, n)` code for `n >= 4` and chooses a nearest codeword. Enumeration is exact and uses a
fixed tie-break: lowest packed codeword. At five qubits the search has 64 candidates.

The selected codeword is a zero-everywhere phase polynomial modulo 8. Adding it changes which
coefficients are odd without changing the circuit's phase function. The implementation verifies
this statement directly for every input before synthesis.

## Synthesis and verification

Each nonzero phase coefficient is synthesized in Python with a compute-phase-uncompute parity
gadget. The C++ core synthesizes the terminal linear map by Gaussian elimination over GF(2).

The final Python verifier executes the original and candidate gate lists directly on every basis
input. It does not call extraction, the decoder, or synthesis. It compares both the output bit
string and phase modulo 8.
Failure rejects the candidate; equal or higher T-count returns the original circuit.

## Determinism

Enumeration order, tie-breaking, synthesis order and JSON encoding are fixed. Reports exclude
timestamps, paths and runtime measurements. Equal inputs therefore produce equal circuits and
reports.

## Reference

- Amy and Mosca, *T-count optimization and Reed–Muller codes*, IEEE Transactions on Information
  Theory 65(8), 2019, [doi:10.1109/TIT.2019.2906374](https://doi.org/10.1109/TIT.2019.2906374).
