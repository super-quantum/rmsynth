# 2. Architecture of the library

rmsynth is built in two layers: a fast native core that knows how to decode Reed–Muller codes and estimate T‑depth, and a Python layer that understands quantum circuits, phase polynomials, and all the ergonomics.



At the very bottom sits the compiled C++ extension, exposed to Python as the module `rmcore`. This is where the heavy lifting happens: compact bit‑vector operations, generator matrix construction, several families of RM decoders (Dumer, list decoders, RPA), algebraic normal form (ANF) computation from truth tables, puncturing/un‑puncturing logic for $\mathrm{RM}(r,n)$ vs $\mathrm{RM}(r,n)^*$, and a T‑depth scheduler based on coloring a conflict graph of parity gadgets. The functions exported by `rmcore` are intentionally low‑level. They operate on packed bytes and small integers, and leave all notions of “T‑count”, “phase polynomial”, or “circuit” to the Python side.

On top of that lives the Python package `rmsynth`. The central module here is `core.py`, which defines a lightweight in‑memory representation of a CNOT+phase circuit (`Circuit` and `Gate`), routines for extracting phase polynomials from circuits, and utilities for converting between sparse coefficient dictionaries and dense vectors. `core.py` also contains a reference implementation of RM generator rows and a brute‑force decoder for tiny instances.

Because generator rows and monomial lists are reused in many places, `rm_cache.py` wraps the RM matrix construction functions in a small LRU cache. It becomes the canonical source of “the generator matrix for $\mathrm{RM}(r,n)$ in punctured form, and the corresponding list of monomials”. This ensures that all modules agree on both the content and ordering.

The decoding logic itself is orchestrated by `decoders.py`. This module presents a single entry point, `decode_rm`, which takes a binary word, RM parameters $(n, r)$, a strategy name, and a bundle of options. Depending on the strategy, it may:

* call the C++ Dumer or Dumer‑List decoders (`decode_rm_dumer_punctured`, `decode_rm_dumer_list_punctured`, `decode_rm_dumer_list_chase_punctured`, `decode_rm_rpa1_punctured`) via `rmcore`;
* invoke a pure‑Python brute‑force ML decoder for small $n$;
* perform $\mathrm{GL}(n,2)$ preconditioning by randomly changing the monomial basis, decoding there, and transforming back;
* apply local exact or branch‑and‑bound “SNAP” refinements around a baseline codeword;
* or run Ordered Statistics Decoding (OSD), implemented in `osd.py`, to explore a small neighborhood of the current solution.

All these paths ultimately return the same kind of result: a punctured codeword (`code_bits`), a set of selected monomials (`selected_monomials`), and a Hamming distance (`dist`). A small policy function `policy_decide` is responsible for resolving “which candidate is better?” when multiple strategies are combined, and it can operate in pure distance mode, or in a mixed distance+depth mode that consults T‑depth estimators from `rmcore`.

The user‑facing abstraction over all of this is the `Optimizer` class in `optimizer.py`. An `Optimizer` is configured with a decoder name (`"auto"`, `"dumer"`, `"rpa"`, `"rpa2"`, `"osd2"`, …), an effort level, and optional knobs like list size, RPA iterations, or a policy for trading off distance and depth. Its `optimize` method takes a `Circuit`, extracts the phase polynomial, chooses an appropriate decoding strategy, runs that strategy via `decoders.decode_rm`, checks algebraic contracts, lifts the chosen monomials back to an 8‑bit phase correction, applies that correction to the coefficients, resynthesizes a circuit, and returns the new circuit plus an `OptimizeReport` with all the important metrics.

The module `contracts.py` sits quietly to the side and provides safety rails. It encodes three key invariants: that every selected monomial really lies in $\mathrm{RM}(r,n)$, that the reported codeword equals the XOR of the generator rows for those monomials, and that the reported distance matches the T‑count of the “after” coefficients. By default the optimizer runs a lighter “maybe check” version of these contracts, gated by an environment variable, but you can enable strict checking while developing.

For general $R_Z(2\pi/2^k)$ or $R_Z(2\pi/d)$ circuits, the `rotk.py` module extends the picture from T‑count ($\mathbb{Z}_8$) to multi‑order phase polynomials. It knows how to reinterpret phase coefficients modulo $2^k$ or $d$, slice them into bit‑planes, run RM decoders plane‑by‑plane in the correct $\mathrm{RM}(r,n)$ parameters, and reassemble the result. It also provides a composite‑$d$ wrapper that uses the Chinese Remainder Theorem to treat the 2‑power and odd parts of $d$ separately.

There are a few more supporting pieces. 
* `autotune.py` is a calibration helper: given a target latency per decode and rough information about the T‑count before optimization, it runs small synthetic workloads through the decoder and chooses beam sizes and RPA iterations that hit the latency target. 
* `bench.py` contains a small benchmarking harness for synthetic $\mathbb{Z}_8$ vectors with different structures (near‑constant oddness, random sparse, random $\mathbb{Z}_8$). 
* `cli.py` exposes a command‑line tool 
* `rmsynth-optimize` lets you generate or load coefficient vectors, call the optimizer, and inspect the result without writing Python. Finally, `rmcore.py` is a dynamic loader that searches for a compiled `rmcore` binary in the package and on `sys.path`, then imports it under the consistent name `rmsynth.rmcore`.

Overall, the architecture is deliberately simple: the Python layer handles all the semantic interpretation (circuits, phase polynomials, RM parameters, policies), and the C++ core is a collection of fast decoders and schedulers that operate on raw bits.

## 2.2 Data representations

Most of the subtlety in the library comes down to data representation choices and making sure they’re consistent:

### 1. Bit and index conventions

Binary words are always in little‑endian order. For a list `bits` of length $L$, bit `bits[i]` is the coefficient attached to the monomial `order[i]`, where `order` is the list of masks produced by `mk_positions(n)`:

```python
order = [1, 2, 3, ..., 2^n - 1]
```
Column `i` in a generator row corresponds to the input point with mask `order[i]` (a nonzero point in $\{0,1\}^n$). The first coordinate of the full RM code (the all‑zero input) is omitted throughout, everything is expressed in terms of a length $L = 2^n-1$ punctured representation.

Monomials themselves are represented as integer masks in $[1, 2^n-1]$: the `i`-th bit in the mask is set if the monomial contains variable $x_i$. The degree of a monomial is simply the population count of that mask.

### 2. BitVec vs Python ints and lists

On the C++ side, the type `BitVec` holds a fixed‑length vector of bits backed by a `std::vector<uint64_t>`. It exposes operations `like get`, `set1`, `xor_inplace`, `slice`, `concat`, and `weight()`. These are used for all recursive decoding and list decoding routines and are never exposed directly to user code.

At the Python layer, words are represented either as:
* lists of bits, e.g. `List[int]` with values 0 or 1, or
* integers where bit $i$ represents coordinate $i$ (mostly in the pure‑Python reference implementations).

Helper functions like `pack_bits_le` and `unpack_bits_le` convert between bit lists and byte strings in little‑endian order so that Python can pass words to `rmcore` and recover them.

### 3. Phase coefficients: dicts vs vectors

Phase polynomials over $\mathbb{Z}_8$ or $\mathbb{Z}_{2^k}$ appear in two forms:
* as sparse dictionaries `{mask: coeff}`, usually the direct output of `extract_phase_coeffs`, which are easy to read and manipulate, and
* as dense vectors `vec` of length $2^n-1$, which are easier to decode and to feed into bit‑plane logic.

`coeffs_to_ve` and `vec_to_coeffs` handle the round‑trip between these forms by agreeing on the canonical monomial order from `mk_positions`. At the multi‑order level, `_as_vec` and `_to_dict` in `rotk.py` generalize this to arbitrary moduli, always taking care to reduce coefficients modulo $2^k$ or $d$ consistently.

### 4. Full vs punctured truth tables

The C++ decoders sometimes work on full truth tables of length $2^n$, particularly in the RPA and Dumer recursive routines, which use the classic Plotkin decomposition. In those situations, the missing “all‑zero input” bit is temporarily reintroduced via `embed_punctured_to_full`, which adds a synthetic first coordinate, and then removed again via `puncture_full` after decoding. The Python layer never sees these full tables; from its perspective everything is length $2^n-1$.

With these conventions in place, rmsynth can treat “phase coefficients as a vector” and “oddness as a bit list” as stable notions, and all the decoders and schedulers can plug into that picture without worrying about off‑by‑one errors or mismatched orderings.