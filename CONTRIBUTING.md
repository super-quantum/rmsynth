# Contributing

Bug reports and small, well-tested improvements within the documented scope are welcome. Proposals
for new gate families, heuristic decoders, native extensions or larger optimization limits should be
discussed in an issue before implementation.

## Development setup

```console
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the local checks before opening a pull request:

```console
ruff check .
ruff format --check .
mypy src
pytest --cov
cmake -S . -B build/native -DRMSYNTH_BUILD_PYTHON=OFF -DBUILD_TESTING=ON \
  -DRMSYNTH_WARNINGS_AS_ERRORS=ON
cmake --build build/native
ctest --test-dir build/native --output-on-failure
python tools/check_public_repo.py
python tools/check_links.py
python -m build
twine check dist/*
check-wheel-contents dist/*.whl
python tools/check_dist.py dist/*
```

Tests should be deterministic, require no network access, and finish comfortably within the existing
ten-second per-test limit. Correctness tests should use an independent oracle rather than restating
the production implementation.

C++ changes must stay within the supported native surface, compile as C++20 without extensions, and
pass the warning-as-error build on GCC, Clang, and MSVC. Format native files with `clang-format`
using the repository configuration.

Contributions are licensed under Apache-2.0. Do not submit confidential code, data, credentials,
generated build output or material that you do not have the right to contribute.
