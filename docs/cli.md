# 11. CLI and integration

## 11.1 CLI tool (`cli.py`)

For quick experimentation without writing Python, the package ships a command-line tool:

```bash
rmsynth-optimize
```

The CLI is essentially a thin wrapper over `Optimizer`. It supports two ways of specifying the phase vector:

1.  Provide `--vec-json path/to/file.json`, where the JSON file contains a list of integers (a $\mathbb{Z}_8$ vector of length $2^n - 1$).
2.  Or ask it to generate a synthetic vector in-process using `--gen` and related options.

Key arguments:

* `--decoder` – one of `dumer`, `dumer-list`, `rpa`, `ml-exact`, or `auto` (default). These are passed directly to the `Optimizer` constructor.
* `--effort` – integer effort level (default 3) that controls beam size and RPA parameters, just like in the Python API.
* `--n` – number of qubits (default 6).
* `--vec-json` – path to a JSON file with a list of integers.
* `--gen` – synthetic generator: `near1`, `rand_sparse`, or `rand_z8`.
* `--flips`, `--density`, `--seed` – parameters for the synthetic generators.
* `--json` – optional path to a JSON report file.

The core logic is:

1.  Load or generate a `vec` via `gen_vec`.
2.  Build the original circuit with `synthesize_from_coeffs(vec, n)`.
3.  Construct an `Optimizer(decoder=args.decoder, effort=args.effort)`.
4.  Call `opt.optimize(circ)` to get `new_circ`, `rep`.
5.  Print:
    * Decoder, effort, n.
    * T-before, T-after, distance.
    * The actual decoder strategy used internally (`opt.last_decoder_used`), which matters if `decoder="auto"`.

If `--json` is provided, it writes a JSON object with basic fields:
```json
{
  "decoder": "...",
  "effort": 3,
  "n": 6,
  "before_t": 32,
  "after_t": 18,
  "distance": 14,
  "strategy": "...",
  "selected_monomials": [...],
  "signature": "..."
}
```
This format is intentionally simple so it can be ingested into scripts or dashboards.

## 11.2 rmcore loader (`rmcore.py`)

While `decoders._load_rmcore()` tries to find the compiled extension in several ways, `rmcore.py` provides a more structured import mechanism for the package namespace `rmsynth.rmcore`.

The logic is:

**`_find_binary_path()`:**

* Inspects the package directory (`os.path.dirname(__file__)`) for files whose names start with `rmcore` and end with any of the extension suffixes known to Python (`.so`, `.pyd`, etc.).
* Scans each directory in `sys.path` for a `rmsynth` subdirectory and looks there for such files.
* Deduplicates candidates and sorts them by modification time, choosing the newest one.

**`_load_as_rmcore(path)`:**

* Creates an `ExtensionFileLoader` and `ModuleSpec` for the name `"rmsynth.rmcore"`.
* Loads and executes the binary extension from `path`, effectively calling its `PyInit_rmcore`.

Finally, at import time:

1.  `_path = _find_binary_path()` is evaluated.
2.  If `_path` is `None`, an `ImportError` is raised with a clear message:
    “Compiled rmcore extension not found … Please build/install the project first, e.g.: `pip install -v .`”
3.  If `_path` is found, `_mod = _load_as_rmcore(_path)` is created, and `sys.modules[__name__]` is replaced with that module. This means that importing `rmsynth.rmcore` gives direct access to the compiled functions, just like a normal extension.

This loader is complementary to `_load_rmcore()` in `decoders.py`, which tries several import patterns but falls back gracefully to pure‑Python decoding if the extension cannot be found. Together they allow both “installed package” and “local build” workflows.