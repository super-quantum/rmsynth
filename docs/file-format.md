# File format and CLI

Circuit JSON uses the schema identifier `rmsynth-reference/circuit-v1` and exactly three top-level
fields: `schema`, `qubits`, and `operations`. Unknown fields are rejected.

## Gates

```json
{"gate": "cnot", "control": 0, "target": 1}
```

```json
{"gate": "phase", "qubit": 0, "exponent": 1}
```

Qubit indices start at zero. Phase exponents are canonical integers from 0 through 7. Boolean values
are not accepted as integers.

## Commands

`rmsynth-ref inspect INPUT` prints a one-line JSON summary.

`rmsynth-ref optimize INPUT --output OUTPUT --report REPORT` writes the optimized circuit and a
deterministic report. Existing files are refused unless `--force` is present.

`rmsynth-ref verify EXPECTED ACTUAL` prints an equivalence result. An inequivalent result includes
the first computational-basis witness.

Use `-` as the input path to read a circuit from standard input. Diagnostics are written to standard
error.

## Exit status

| Status | Meaning |
|---:|---|
| 0 | Success or equivalent circuits |
| 2 | Invalid input or command usage |
| 3 | Resource limit exceeded |
| 4 | Inequivalence or internal verification failure |
