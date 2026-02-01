# rmsynth

`rmsynth` is a resource-optimization toolkit for **Clifford+T** quantum circuits. It represents circuits as **phase polynomials** and applies **punctured Reed–Muller decoding** to reduce **T-count** and (optionally) **T-depth**. The project provides a Python API and CLI backed by a high-performance C++ extension (`rmcore`).

## Features

- Phase-polynomial extraction/synthesis for CNOT+phase circuits
- Multiple decoding backends (Dumer, list/chase, RPA, OSD) with deterministic tie-breaking options
- Depth-aware selection and optional scheduling
- Autotuning of decoder parameters
- Contracts/tests to validate decoding correctness and invariants

## Local installation

### Prerequisites

- Python 3.9+ (3.10/3.11 recommended)
- A C++20 compiler (Apple Clang / GCC / MSVC)
- CMake 3.18+
- `pip`, `setuptools`, `wheel`

### Install (editable)

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

## License

License: Apache-2.0
