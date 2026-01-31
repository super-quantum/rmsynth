# 3. Core phase–polynomial toolkit (core.py and friends)

The `core.py` module is where “quantum circuit” meets “coding theory”. It defines a small circuit model, the machinery to extract phase polynomials, the mapping to and from vectors suitable for RM decoding, a Python reference implementation of RM generator rows and a brute‑force decoder, and synthesis back to circuits from optimized coefficients. Most users never call these functions directly, but understanding them makes the rest of the library much easier to follow.

## 3.1 Circuits and gates

The circuit model is intentionally spartan: the only gate types are CNOT and a generic Z‑phase gate with exponent modulo 8. This matches the Amy–Mosca setup, where arbitrary CNOT networks plus $Z^{(k)}$ phases generate the class of linear‑phase Clifford+T circuits.

Each gate is represented by an instance of `Gate`, with a `kind` string and a handful of optional fields:

* for CNOTs: `kind="cnot"`, with `ctrl` and `tgt` set to the control and target qubit indices;
* for phase gates: `kind="phase"`, with `q` denoting the target qubit and `k` the integer exponent of the phase (always stored modulo 8).

The `Circuit` class is a simple container around a list of these gates and a qubit count:

```python
class Circuit:
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.ops: List[Gate] = []

    def add_cnot(self, ctrl: int, tgt: int):
        assert 0 <= ctrl < self.n and 0 <= tgt < self.n and ctrl != tgt
        self.ops.append(Gate("cnot", ctrl=ctrl, tgt=tgt))

    def add_phase(self, q: int, k: int):
        self.ops.append(Gate("phase", q=q, k=k % 8))

    def t_count(self) -> int:
        return sum(1 for g in self.ops if g.kind == "phase" and (g.k & 1))
```

`add_cnot` and `add_phase` are helpers used by synthesis and by tests. `t_count` is a quick “visible” T‑count, which counts how many phase gates have odd exponent. This may differ slightly from the T‑count implied by the phase polynomial if there are hidden cancellations, but once you re‑synthesize from optimized coefficients the two coincide.

This representation deliberately ignores Clifford‑only gates like H. The intended usage is to start from circuits that have already been normalized into a CNOT+$Z^{(k)}$ form, or to treat CNOT+phase layers inside a larger compiler flow.

## 3.2 Extracting and manipulating coefficients

The core transformation from circuit to phase polynomial is implemented by `extract_phase_coeffs`. Its job is to simulate how CNOTs transform the logical parities on each qubit and to record how much phase is applied to each parity.

At the start, each qubit $j$ carries the “basis” variable $x_j$. We represent the logical form on each qubit as an integer mask, with a single bit set to indicate the variable:

```python
forms = [1 << i for i in range(n)]
```

As we scan the circuit:

1.  When we see a CNOT from `c` to `t`, we update the target’s form to reflect the new parity $x_t \oplus x_c$. In mask form, XORing the masks does exactly that:
    ```python
    forms[t] ^= forms[c]
    ```
2.  When we see a phase gate on qubit `q` with exponent `k`, the current form `forms[q]` is exactly the monomial mask whose parity is being phased. We look up the old coefficient for that mask in a dictionary, add `k` modulo 8, and store it back.

Putting that together:

```python
def extract_phase_coeffs(circ: Circuit) -> Dict[int, int]:
    n = circ.n
    forms = [1 << i for i in range(n)]
    a: Dict[int, int] = {}
    for g in circ.ops:
        if g.kind == "cnot":
            c, t = g.ctrl, g.tgt
            forms[t] ^= forms[c]
        elif g.kind == "phase":
            mask = forms[g.q]
            a[mask] = (a.get(mask, 0) + (g.k % 8)) % 8
        else:
            raise ValueError("unknown gate")
    return a
```

The result is a map `a` from monomial masks to coefficients in $\mathbb{Z}_8$. This is the algebraic object that the coding‑theoretic part of the library works with.

To feed this into a decoder, we need to pack it into a vector in a fixed monomial order. The helper `mk_positions(n)` establishes that order as all nonzero masks in ascending numeric order, and provides a reverse lookup:

```python
order = [1, 2, 3, ..., 2^n - 1]
pos = {mask: index}
```

Using this, `coeffs_to_vec` and `vec_to_coeffs` implement the conversion between sparse maps and dense vectors. `coeffs_to_vec` constructs a length‑$L$ vector and places each coefficient at the position determined by `pos[mask]`. `vec_to_coeffs` runs this in reverse, skipping zeros and mapping indices back to masks through `order`.

With coefficient vectors in hand, a few small helpers provide basic arithmetic:

* `res2_vec(vec)` takes the least significant bit of each coefficient and packs them into an integer bitmask; this is the oddness vector that we decode.
* `add_mod8_vec(a, b)` adds two coefficient vectors element‑wise modulo 8.
* `t_count_of_coeffs(vec)` counts how many coefficients are odd, i.e. how many T‑like terms are present.

These helpers are used directly by the optimizer and multi‑order routines and indirectly by the contracts.

## 3.3 RM generator and brute‑force decoding (Python reference)

While the high‑performance decoders live in C++, the Python codebase keeps a complete “reference implementation” of RM generator matrices and a brute‑force decoder.

For fixed $n$ and $r$, `rm_generator_rows(n, r)` constructs the generator rows for $\mathrm{RM}(r,n)^*$ in the monomial order that the rest of the Python code uses. It begins by building the list of monomials of degree at most $r$ (including the constant monomial if $r \ge 0$), then, for each monomial $t$, evaluates whether $t$ divides each nonzero mask $y$ in order. If $t$ divides $y$, the monomial evaluates to 1 at that point and we set the corresponding bit in the row.

The result is a list of Python integers, each representing a row in a $K \times L$ generator matrix ($K$ monomials, $L = 2^n – 1$ positions). Bit $j$ of row $i$ indicates whether monomial $i$ evaluates to 1 at input $j$.

The dimension function `rm_dimension(n, r)` computes $\sum_{d=0}^r \binom{n}{d}$. It’s used both for reporting (e.g. in `DecodeReport`) and by some of the pure‑Python routines that need to know how many monomials are in the basis.

The brute‑force decoder `decode_rm_bruteforce` sits on top of these rows. It takes as input a binary word `w_bits` represented as an integer, loops over all possible choices of monomial coefficients $u$ in $\mathrm{GF}(2)$ (from 0 up to $2^K - 1$), forms the corresponding codeword by XORing the generator rows where $u$ has a 1, and measures Hamming distance to `w_bits`. It tracks the best codeword found, the corresponding coefficient vector $u$, and how many ties it encountered at the best distance. Once the search terminates, it converts $u$ back into a list of monomials and returns:

1.  the best codeword `cw` as an integer,
2.  the list of selected monomials,
3.  and the number of ties.

## 3.4 Synthesis from coefficients

The final step of the pipeline is to go back from a phase‑polynomial vector to an actual circuit. This is handled by `synthesize_from_coeffs`, which takes a vector of coefficients `vec`, a qubit count `n`, and optional scheduling information, and returns a `Circuit`.

The basic idea is to implement each nonzero monomial as a **parity gadget**:

1.  Choose a target qubit as the least significant set bit in the monomial mask.
2.  For each other qubit in the mask, apply a CNOT into the target to accumulate the parity of those bits onto the target.
3.  Apply a phase gate with exponent equal to the monomial coefficient on the target.
4.  Undo the CNOTs in reverse order to restore the original logical state.

In code, a private helper `_emit_monomial(circ, mask, k)` does exactly this. It takes care of quickly scanning the set bits of `mask`, doing the forward and backward CNOT ladders, and skipping monomials whose coefficient is zero modulo 8.

The `synthesize_from_coeffs` function starts by converting the input vector into a dictionary of nonzero coefficients with `vec_to_coeffs`. If the dictionary is empty, it returns an empty `Circuit`. Otherwise, it behaves in one of two modes:

* **Sequential mode (the default):** if no schedule is supplied and `use_schedule` is false, the function simply iterates through the canonical monomial order and emits a parity gadget for every nonzero coefficient. This reproduces the “naïve” synthesis from the original Amy–Mosca picture.
* **Depth‑aware mode:** if you pass explicit layers, or set `use_schedule=True`, the function changes how it orders monomials. In the `use_schedule` case it attempts to load the `rmcore` extension and call `tdepth_schedule_from_monoms` on the set of odd monomials. The scheduler returns a depth and a list of layers, where each layer is a list of monomial masks that can be executed in parallel without conflict. `synthesize_from_coeffs` then:
    1.  Emits all T‑like monomials in layer order, respecting the layer boundaries.
    2.  Emits any remaining monomials (such as those with even coefficients) afterwards, in a stable deterministic order.

This design keeps the synthesis function flexible. For quick T‑count optimization you can ignore scheduling entirely and keep everything sequential, but if you care about T‑depth you can either compute layers separately with `compute_tdepth_layers_from_vec` or let `synthesize_from_coeffs` call into the scheduler for you.

On top of these core functions, `core.py` also offers a one‑liner `synthesize_with_schedule` that wraps the common pattern “take a coefficient vector, compute layers, synthesize accordingly, and report the depth and layer structure”. This is convenient when you want to look at T‑depth directly from coefficients, without invoking the whole optimizer pipeline.

Finally, `rm_cache.py` ties back into this picture by ensuring that all code that ever builds RM generator rows or monomial lists does so in a consistent, cached, and shared way. Whether you are decoding in Python, calling into C++ from decoders, running OSD, or verifying invariants in contracts, the underlying basis and generator matrix are the same, and all of the phase‑polynomial machinery in `core.py` can rely on that.