# RMSynth Reference Edition

[![CI](https://github.com/super-quantum/rmsynth/actions/workflows/ci.yml/badge.svg)](https://github.com/super-quantum/rmsynth/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

RMSynth Reference Edition is a small implementation of T-count optimization for CNOT-phase
circuits. A compact C++20 core performs exact decoding and GF(2) synthesis; Python provides the
public API, file handling, and an independent verifier.

This repository is not the source code for other RMSynth products and does not aim for feature or
performance parity with them. If you are interested in up-to-date optimization capabilities, reach
out to us directly.

## Scope

The optimizer accepts CNOT gates and diagonal gates `diag(1, exp(iπk/4))`, where `k` is an integer
from 0 to 7. It supports at most five qubits. Every changed circuit is checked on
all computational-basis inputs before it is returned, and an optimization is accepted only when it
strictly lowers T-count.

Circuit inspection and equivalence checking support up to ten qubits. T-depth optimization,
arbitrary Clifford gates, native extensions and production-scale workloads are outside the scope of
this edition. See [Scope and limits](docs/scope.md) for the complete contract.

## Install

The current version is a release candidate. Building from source requires Python 3.11 or newer and
a C++20 compiler. Install a local checkout with:

```console
python -m pip install .
```

Development tools are optional:

```console
python -m pip install -e '.[dev]'
```

The build uses CMake through `scikit-build-core`. The installed package has no third-party runtime
dependency.

## Python example

```python
from rmsynth_reference import (
    LinearMap,
    LinearPhaseProgram,
    PhasePolynomial,
    optimize,
    synthesize_program,
)

program = LinearPhaseProgram(
    PhasePolynomial(4, (1,) * 15),
    LinearMap.identity(4),
)
circuit = synthesize_program(program)
result = optimize(circuit)

print(result.report.before_t_count, result.report.after_t_count)
# 15 0
```

The complete example is in [examples/all_parities.py](examples/all_parities.py).

## Command line

Circuit files use a versioned JSON format:

```json
{
  "schema": "rmsynth-reference/circuit-v1",
  "qubits": 1,
  "operations": [
    {"gate": "phase", "qubit": 0, "exponent": 1},
    {"gate": "phase", "qubit": 0, "exponent": 1}
  ]
}
```

```console
rmsynth-ref inspect examples/cancellation.json
rmsynth-ref optimize examples/cancellation.json --output optimized.json --report report.json
rmsynth-ref verify examples/cancellation.json optimized.json
```

The optimizer never overwrites a file unless `--force` is supplied. Details are in
[File format and CLI](docs/file-format.md).

## Correctness

The implementation extracts the phase polynomial and final GF(2) linear map, decodes a bounded
punctured Reed–Muller instance exactly in C++, synthesizes a candidate, and checks the result with a
separate Python gate evaluator. The test suite also exhausts every four-qubit decoder input and all
invertible 3×3 binary maps. See [Design and correctness](docs/design.md).

## Development

```console
ruff check .
ruff format --check .
mypy src
pytest --cov
cmake -S . -B build/native -DRMSYNTH_BUILD_PYTHON=OFF -DBUILD_TESTING=ON
cmake --build build/native
ctest --test-dir build/native --output-on-failure
python -m build
python tools/check_dist.py dist/*
```

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing a change. Report security issues as
described in [SECURITY.md](SECURITY.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
