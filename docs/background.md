# 1. Background and theory

This section explains the mathematical background that underpins the entire library, and how it relates to the Amy–Mosca framework.

## 1.1 Linear‑phase circuits and phase polynomials

A linear‑phase circuit on $n$ qubits is one that is diagonal in the computational basis and whose phase on basis state $|x\rangle$ depends only on linear and higher‑order parities of the bits of $x$.

For Clifford+T circuits over $\{ \text{CNOT}, T, S, Z \}$, Amy & Mosca show that any such unitary can be written as:

$$
U_f : |x\rangle \mapsto \exp\left( i \frac{\pi}{4} f(x) \right) | \pi(x)\rangle
$$

where:

* $\pi$ is a linear permutation (implemented by CNOTs).
* $f(x)$ is an integer‑valued polynomial in bits $x = (x_0,\dots,x_{n-1})$, formed from monomials $x^m = \prod_{i\in S} x_i$ with integer coefficients.

In our implementation:

* The linear permutation $\pi$ is handled implicitly by the CNOT network in the circuit representation.
* The phase component is captured by a phase polynomial

$$
  f(x) = \sum_{m} a_m x^m, \quad a_m \in \mathbb{Z}_8
$$

  where:
    * Each monomial corresponds to a mask $m \in \{1, \dots, 2^n-1\}$.
    * Coefficients $a_m$ are taken modulo 8 for Clifford+T (because $T^8 = \mathbf{1}$).

In the code:

* `core.Circuit` represents a CNOT+phase circuit.
* `core.extract_phase_coeffs(circ)` walks through the circuit:
    * Maintains a list of current linear forms representing the effect of CNOTs.
    * For each phase gate on qubit $q$ with exponent $k$, determines the mask corresponding to that qubit’s current linear form and accumulates $+k\bmod 8$ into a dict `a[mask]`.
* `core.coeffs_to_vec(a, n)` and `vec_to_coeffs(vec, n)` convert between:
    * Sparse maps `{mask -> coeff}`, and
    * Dense vectors indexed by a fixed order of monomials (`mk_positions(n)`).

This gives us a pure algebraic representation of the phase part of the circuit as a vector $a \in \mathbb{Z}_8^{2^n-1}$, independent of any particular gate sequence.

### Oddness and T‑count

The T‑count is the number of T‑like phases in the circuit, where “T‑like” means a phase gate with odd coefficient in $\mathbb{Z}_8$. Formally:

$$
\text{T‑count}(a) = \sum_m (a_m \bmod 2).
$$

We define the oddness vector

$$
w = \operatorname{Res}_2(a) \in \{0,1\}^{2^n-1}, \quad w_m = a_m \bmod 2.
$$

This vector is precisely what we feed into the RM decoders.

Choosing a different set of monomials (a different “basis” of parity gadgets) but describing the same unitary corresponds to adding a codeword in a certain RM code to this vector. This is the key insight of the Amy–Mosca paper, and the reason coding theory enters the picture.

## 1.2 Reed–Muller codes and monomial structure

The binary Reed–Muller code $\mathrm{RM}(r,n)$ consists of evaluation vectors of multilinear Boolean polynomials of degree $\le r$ in $n$ variables, over all $2^n$ inputs:

$$
\mathrm{RM}(r,n) = \{ (f(x))_{x \in \{0,1\}^n} : \deg(f) \le r, f : \{0,1\}^n \to \mathbb{F}_2 \}.
$$

In the Amy–Mosca construction, and in this codebase, we work with the punctured code $\mathrm{RM}(r,n)^*$ obtained by omitting the all‑zero input ($x=0$).

### Generator matrix from monomials

A natural generator for $\mathrm{RM}(r,n)^*$ is:

* Rows indexed by monomials $x^m$ with degree $\le r$ (including the constant monomial).
* Columns indexed by nonzero inputs $y \in \{0,1\}^n \setminus \{0\}$.
* Entry $G_{m,y} = x^m(y)$ evaluated modulo 2.

In code:

* The C++ `rm_generator_rows(n, r)` and Python `core.rm_generator_rows(n, r)` both implement this mapping:
    * `monoms` is the list of monomial masks $m$ with $\deg(m) \le r$ (and optionally $m=0$)
    * Each row is a word of length $L = 2^n-1$ whose bits correspond to whether $x^m(y) = 1$ at each nonzero input $y$
* We maintain a consistent monomial ordering across:
    * `rm_generator_rows` in C++
    * `core.rm_generator_rows` in Python
    * `rm_cache.get_rows` and `rm_cache.get_monom_list`
    * The contracts in `contracts.py`

This alignment is critical: it ensures that “adding monomials” in the coefficient space is exactly the same as adding generator rows in the RM code.

### Reed–Muller parameters relevant here

Some basic facts we use (but don’t re‑prove):

* $\mathrm{RM}(r,n)$ has:
    * Length $N = 2^n$.
    * Dimension $\sum_{d=0}^{r} \binom{n}{d}$. (Error Correction Zoo)
* $\mathrm{RM}(n-4,n)^*$ is the punctured code for degree $\le n-4$ polynomials; this is the one that appears in the T‑count equivalence.
* $\mathrm{RM}(n-k-1,n)^*$ appears for bit‑plane $k$ in more general rotations $R_Z(2\pi / 2^k)$.

The code itself doesn’t rely on specific minimum distances or weight distributions. It treats RM codes as ordinary linear codes with an explicit generator matrix, and plugs in classical decoders that know how to exploit the RM structure.

## 1.3 From T‑count optimization to RM decoding

The main theorem of Amy–Mosca (stated informally) is:

"Minimizing the T‑count of an $n$-qubit unitary over $\{\text{CNOT}, T\}$ plus Clifford powers of $T$ is polynomially equivalent to minimum‑distance decoding of a length $2^n-1$ binary vector in the punctured Reed–Muller code $\mathrm{RM}(n-4,n)^*$."

More concretely:

1.  Given a phase polynomial $f(x)$ over $\mathbb{Z}_8$, collect coefficients $a_m$.
2.  Define the oddness vector $w = \operatorname{Res}_2(a)$.
3.  Let $C = \mathrm{RM}(n-4,n)^*$. Each codeword $c \in C$ corresponds to a valid way to re‑express the unitary using a different set of T‑like gates, because:
    * Adding $c$ corresponds to adding a degree $\le n-4$ Boolean polynomial to the phase function modulo 2.
    * This is exactly adding a phase polynomial that (up to Clifford gates) implements the identity.
4.  Choose $c^\star \in C$ to minimize the Hamming distance: $c^\star = \arg\min_{c \in C} d_H(w, c).$

5.  Define a correction vector $d = w \oplus c^\star$. The positions of 1s in $d$ correspond to monomials that remain T‑like in the optimized circuit.
6.  Lift $d$ into $\mathbb{Z}_8$ coefficients (e.g., as a correction phase polynomial $g(x)$) and add to the original coefficients:

$$
    a' = a + g \pmod{8}.
$$

T‑count of the new phase polynomial is exactly $|d|$, the minimal distance.

In the code:

* `core.res2_vec(vec)` computes $w$.
* C++/Python RM decoders implement various approximations to

$$
\arg\min_{c \in \mathrm{RM}(n-4,n)^*} d_H(w,c).
$$

  `decoders.decode_rm(...)` is a strategy switch that picks a particular decoder, ranging from exact brute‑force ("ml-exact") to advanced heuristics ("rpa-adv", "rpa2", OSD variants, GL(n,2) search).
* `core.lift_to_c(selected_monomials, n)` and similar functions build a phase correction vector from a set of monomials, corresponding to the chosen codeword.
* `core.add_mod8_vec(vec, c_vec)` produces the optimized coefficients.

### Contracts and correctness checks

To ensure that decoders respect this structure, `contracts.py` defines three key conditions:

1.  **Monomials lie in RM(r,n)**
    All selected monomials must have degree $\le r$. For T‑count, $r = n-4$.
2.  **Codeword consistency**
    The punctured codeword bits `code_bits` returned by the decoder must equal the XOR of generator rows corresponding to `selected_monomials`.
3.  **Distance equals T‑count**
    The reported distance (Hamming weight of the residual after decoding) must equal the T‑count of the “after” coefficients obtained by applying the correction.

These contracts ensure that, no matter which decoder strategy is chosen (Dumer, RPA, OSD, etc.), the result is mathematically consistent with the theory, even when the decoder is approximate (i.e., not ML).

## 1.4 Small‑angle rotations and multi‑order optimization

Amy & Mosca also show that the Reed–Muller connection extends to more general small‑angle rotations of the form $R_Z(2\pi / d)$.
Let $d$ be a positive integer, and write it as:

$$
d = 2^k d_{\text{odd}},
$$

where $d_{\text{odd}}$ is odd and $k = v_2(d)$ is the 2‑adic valuation.
Then:

* The phase coefficients $a_m$ live in $\mathbb{Z}_d$.
* The least significant bit‑plane of $a_m$ (mod 2) again encodes T‑like gates.
* Higher bit‑planes correspond to coarser rotations $R_Z(2\pi / 2^k)$.

Amy–Mosca show that minimizing the number of $R_Z(2\pi/d)$ gates is equivalent to MW decoding in $\mathrm{RM}(n-k-1,n)^*$, using a suitable mapping of coefficients to binary vectors.

### Bit‑plane decomposition for $2^k$

When $d = 2^k$, we write each coefficient as:

$$
a_m = \sum_{\ell=1}^k 2^{k-\ell} a_m^{(\ell)}, \quad a_m^{(\ell)} \in \{0,1\}.
$$

For each plane $\ell\in \{1,\dots,k\}$, we define a binary word

$$
w^{(\ell)} = (a_m^{(\ell)})_m.
$$

Each plane is associated with an RM code $\mathrm{RM}(n-\ell-1,n)^*$.
Intuitively, higher planes (MSB) correspond to larger angles, and therefore to more “costly” gates.

In our code:

* `rotk._plane_bits_from_vec(a_vec, n, k, ell)` returns $w^{(\ell)}$ as a punctured binary word.
* `rotk.weights_by_plane(a, n, k)` returns a list of weights $|w^{(1)}|, ..., |w^{(k)}|$.

### Multi‑order bit‑plane optimization

The paper describes a multi‑order procedure that optimizes these planes, starting from the most significant plane and working downwards, making sure that the transformations preserve the overall unitary.
This implementation builds on that by:

1.  For each plane $\ell=1$ (MSB) to $k$ (LSB):
    * Extract $w^{(\ell)}$.
    * Decode $w^{(\ell)}$ in $\mathrm{RM}(n-\ell-1,n)^*$ using any strategy supported by `decoders.decode_rm`.
    * Optionally apply per‑plane OSD and GL(n,2) search to refine the candidate.
    * Update just that plane’s parity pattern by combining the residual with the decoder’s codeword, without affecting other planes (“no‑carry update”).
2.  After processing all planes, recombine the planes into a vector $a' \in \mathbb{Z}_{2^k}^{2^n-1}$ without mixing bits across planes:
    `rotk._rebuild_from_planes(planes, k)`.
3.  Convert back to coefficient dict and synthesize a circuit with `synthesize_from_coeffs`.

**Key implementation choice:**

The library enforces a **no‑inter‑plane carry invariant** during bit‑plane optimization:
Each plane is optimized in a way that never increases that plane’s weight, and does not modify other planes.
This makes the per‑plane statistics stable and easy to reason about.

This is slightly more restrictive than the most general form in the paper (which allows transformations that couple planes, so long as they preserve the overall phase). The tradeoff is:

* Easier reasoning and testing per plane.
* Slightly less aggressive global optimization in corner cases.

This behavior is documented explicitly in `rotk.optimize_multiorder_bitplanes` and is a conscious design decision rather than a limitation of the theory.

### Composite d via CRT

For general $d = 2^k d_{\text{odd}}$, the Amy–Mosca result uses a Chinese Remainder Theorem (CRT) perspective:

* Coefficients $a_m \in \mathbb{Z}_d$ can be represented as pairs $(a_m \bmod 2^k, a_m \bmod d_{\text{odd}})$.
* Optimization can be performed on the 2‑power component (where RM structure is present), while preserving the odd component.

The implementation follows this via:

* `rotk._v2(d)` to compute $k = v_2(d)$.
* `rotk.optimize_multiorder_d(a_in, n, d, ...)`:
    * Extracts the even part `a_even` modulo $2^k$.
    * Runs `optimize_multiorder_bitplanes` on this even part.
    * Recombines with the original odd residues using `_crt_merge_even_odd`, which implements:

$$
      x \equiv a_{\text{even}} \pmod{2^k}, \quad x \equiv a_{\text{odd,ref}} \pmod{d_{\text{odd}}}
$$

and solves for $x \pmod d$ pointwise.

This yields an optimized coefficient vector modulo $d$, preserving the structure mandated by the theory while exploiting the RM codes on the 2‑power side.

## 1.5 T‑depth and parity networks

So far we’ve focused on T‑count (or bit‑plane Hamming weights). In practice, T‑depth is often equally important: it determines how many sequential layers of T‑like gates must be executed, which impacts wall‑clock runtime and error accumulation.

In this library:

* Each nonzero monomial $x^m$ with an odd coefficient corresponds to a **parity gadget**:
    * A ladder of CNOTs that temporarily concentrates the parity of the selected qubits onto a target qubit.
    * A T‑like phase on that target.
    * The inverse ladder to uncompute the parity.
* Two parity gadgets can be performed in parallel (in the same T‑layer) if they do not share qubits. Therefore:

  We build a **conflict graph** whose vertices are monomials $m$ with odd coefficients.
  There is an edge between two monomials $m_i, m_j$ if their masks overlap:

$$
  (m_i \& m_j) \ne 0.
$$

  A proper graph coloring of this graph assigns a color (layer index) to each monomial so that adjacent vertices have different colors.
  The minimal number of colors used is precisely the minimal T‑depth achievable with this parity‑gadget scheme (ignoring re‑ordering of CNOTs).

The provided T‑depth scheduler:

* Encodes this conflict graph as an adjacency bitset and uses a DSATUR‑style branch‑and‑bound algorithm to color it.
* Uses a fast greedy coloring as an upper bound and seed.
* Applies node and time caps (via environment variables) to be robust on larger instances.

This is orthogonal to the Amy–Mosca T‑count → RM decoding equivalence, but fully compatible with it:

1.  You can first use RM decoders to minimize T‑count.
2.  Then run the scheduler to pack the remaining T gates into as few T‑layers as possible.

The decoding front‑end uses a cheap T‑depth estimator (number of residual odd positions) during decoding for scoring candidates. The exact scheduler is called only when the user specifically requests a depth‑aware synthesis or scheduling from monomials.

## 1.6 What rmsynth adds additionally

Relative to the original Amy–Mosca framework, this library:

* Implements the core equivalence (T‑count ↔ decoding in $\mathrm{RM}(n-4,n)^*$) in a way that is visible and testable from the code.
* Extends the framework with:
    * Practical high‑quality decoders (Dumer, Dumer‑list, Chase, RPA, OSD) tuned for small/medium $n$.
    * GL(n,2) preconditioning.
    * Local search heuristics (SNAP, branch‑and‑bound on small pools).
    * Policies for trading off distance vs T‑depth.
* Provides a multi‑order bit‑plane implementation for $R_Z(2\pi/2^k)$ and a CRT‑based wrapper for general $R_Z(2\pi/d)$, with explicit handling of per‑plane invariants.
* Adds a T‑depth scheduler integrated with synthesis.
* Wraps everything into:
    * A user‑friendly Optimizer class.
    * A CLI.
    * Autotuning and benchmarking tools.

The rest of the documentation will build on this background.