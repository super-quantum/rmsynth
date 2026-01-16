from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

# Basic combinatorial helpers

def popcount(x: int) -> int:
    """Hamming weight of an integer."""
    return x.bit_count()

def iter_nonzero_masks(n: int):
    """
    Iterate over all non-zero masks of n bits in increasing order.

    Masks encode monomials x^S for S ⊆ {0,..,n-1}, S ≠ ∅.
    """
    for m in range(1, 1 << n):
        yield m

def weight(m: int) -> int:
    """Alias for popcount, used for monomial degree."""
    return popcount(m)

def mk_positions(n: int):
    """
    Return (order, pos):

      order : list of non-zero masks in increasing order
      pos   : dict mapping mask -> index in 'order'

    This defines the canonical mapping between monomial masks and
    coefficient vector indices used throughout the Python layer.
    """
    order = list(iter_nonzero_masks(n))
    pos = {m: i for i, m in enumerate(order)}
    return order, pos

# Gate / circuit representation (simple Clifford+phase model)

@dataclass
class Gate:
    """
    Simple gate descriptor.

    kind : "cnot" or "phase"
    q    : qubit index (for phase)
    k    : phase exponent k modulo 8 (phase = exp(iπk/4))
    ctrl : control qubit (for CNOT)
    tgt  : target qubit (for CNOT)
    """
    kind: str
    q: Optional[int] = None
    k: Optional[int] = None
    ctrl: Optional[int] = None
    tgt: Optional[int] = None

class Circuit:
    """
    Minimal circuit container: list of Gate objects on n qubits.

    This is not tied to any particular backend, it is just a convenient
    carrier for extract_phase_coeffs / synthesize_from_coeffs.
    """
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.ops: List[Gate] = []

    def add_cnot(self, ctrl: int, tgt: int):
        """Append a CNOT gate with given control and target."""
        assert 0 <= ctrl < self.n and 0 <= tgt < self.n and ctrl != tgt
        self.ops.append(Gate("cnot", ctrl=ctrl, tgt=tgt))

    def add_phase(self, q: int, k: int):
        """
        Append a phase gate on qubit q with exponent k mod 8.

        Convention: phase gate = exp(iπk/4), so k odd corresponds to a T/T†,
        k even to S/Z or identity, depending on k mod 4.
        """
        self.ops.append(Gate("phase", q=q, k=k % 8))

    def t_count(self) -> int:
        """Count gates with odd phase exponent (T-like gates)."""
        return sum(1 for g in self.ops if g.kind == "phase" and (g.k & 1))

# Phase-polynomial extraction

def extract_phase_coeffs(circ: Circuit) -> Dict[int, int]:
    """
    Extract the phase polynomial coefficients a[mask] (mod 8) from a circuit.

    Algorithm:
      - Maintain an array 'forms' of length n where forms[i] encodes the
        current affine linear form for qubit i as a mask over {0..n-1}.
      - A CNOT (c -> t) updates forms[t] ^= forms[c].
      - A phase on qubit q corresponds to adding k to coefficient a[forms[q]].

    Returns:
      dict mapping mask (>=1) -> coefficient in Z8.
    """
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

def coeffs_to_vec(a: Dict[int, int], n: int) -> List[int]:
    """
    Convert sparse coeff dict a[mask] (mod 8) into a dense Z8 vector
    of length (2^n - 1), ordered by mk_positions(n).
    """
    order, pos = mk_positions(n)
    vec = [0]*len(order)
    for y, val in a.items():
        vec[pos[y]] = val % 8
    return vec

def vec_to_coeffs(vec: List[int], n: int) -> Dict[int, int]:
    """
    Inverse of coeffs_to_vec: convert dense vector back to sparse dict.
    """
    order, _ = mk_positions(n)
    return {order[i]: (vec[i] % 8) for i in range(len(order)) if (vec[i] % 8) != 0}

def res2_vec(vec: List[int]) -> int:
    """
    Extract the odd part (LSB) of a Z8 coefficient vector as a bitmask.

    Returns an integer whose i-th bit is 1 iff vec[i] is odd.
    """
    bits = 0
    for i, v in enumerate(vec):
        if v & 1: bits |= (1 << i)
    return bits

def add_mod8_vec(a: List[int], b: List[int]) -> List[int]:
    """Componentwise addition modulo 8 for Z8 vectors."""
    return [(x + y) % 8 for x, y in zip(a, b)]

# RM utilities (pure Python mirrors of rm_code.cpp)

def rm_generator_rows(n: int, r: int) -> List[int]:
    """
    Python version of rm_generator_rows for RM(r,n), punctured length L=2^n-1.

    Returns:
      rows: list of integers, each encoding a generator row as L bits.

    The monomial ordering is:
      [0] (constant) if r>=0, then all masks t with weight(t) <= r.
    """
    order, _ = mk_positions(n)
    rows: List[int] = []
    monoms: List[int] = [0] if r >= 0 else []
    for t in range(1, 1<<n):
        if weight(t) <= r:
            monoms.append(t)
    for t in monoms:
        bits = 0
        for j, y in enumerate(order):
            val = 1 if (t == 0 or (t & y) == t) else 0
            if val: bits |= (1 << j)
        rows.append(bits)
    return rows

def rm_dimension(n: int, r: int) -> int:
    """
    Dimension of RM(r,n): sum_{d=0}^r C(n,d), or 0 for r<0.
    """
    if r < 0: return 0
    from math import comb
    s = 0
    for d in range(0, r+1):
        s += comb(n, d)
    return s

def decode_rm_bruteforce(w_bits: int, n: int, r: int):
    """
    Exhaustive ML decoder for small RM(r,n) instances, in pure Python.

    w_bits : received word as L-bit integer, where L = (2^n - 1)
    n, r   : RM parameters

    Returns (best_word, selected_monomials, ties) with:
      - best_word  : codeword as L-bit int
      - selected   : list of monomial masks corresponding to best_word
      - ties       : number of codewords at minimal distance (for small n)
    """
    if r < 0: return 0, [], 1
    rows = rm_generator_rows(n, r)
    m = len(rows)
    monoms: List[int] = [0] + [t for t in range(1, 1<<n) if weight(t) <= r] if r >= 0 else []
    best_word = None; best_dist = None; best_u = None; ties = 0
    for u in range(1 << m):
        cw = 0
        # XOR rows for monomials where bit is set in u.
        uu = u; idx = 0
        while uu:
            if uu & 1: cw ^= rows[idx]
            uu >>= 1; idx += 1
        if idx < m:
            for j in range(idx, m):
                if (u >> j) & 1: cw ^= rows[j]
        dist = (w_bits ^ cw).bit_count()
        if best_dist is None or dist < best_dist or (dist == best_dist and cw < best_word):
            ties = 1 if (best_dist is None or dist < best_dist) else (ties + 1)
            best_dist = dist; best_word = cw; best_u = u
    selected = [monoms[i] for i in range(m) if (best_u >> i) & 1]
    return best_word, selected, ties

def lift_to_c(selected_monomials: List[int], n: int) -> List[int]:
    """
    Lift a set of selected monomials to a Z8 vector c (mod 8).

    For each monomial t in 'selected_monomials', we add 1 to c[j] for
    all punctured positions j where the monomial evaluates to 1.

    This is the Z8 analogue of forming a codeword from basis rows.
    """
    order, _ = mk_positions(n)
    L = len(order); c = [0]*L
    for t in selected_monomials:
        for j, y in enumerate(order):
            if t == 0 or (t & y) == t: c[j] = (c[j] + 1) % 8
    return c

# Synthesis (phase polynomial -> circuit)

def synthesize_from_coeffs(
        vec: List[int],
        n: int,
        layers: Optional[List[List[int]]] = None,
        *,
        use_schedule: bool = False,
        schedule_budget: int = -1,
) -> Circuit:
    """
    Synthesize a circuit for the phase‑polynomial 'vec' on n qubits.

    (optional, depth-aware):
      - If 'layers' is provided, it must be a list of lists of monomial masks
        (each mask in 1..(2^n-1)). Phase gates are placed layer-by-layer in the
        given order (all gates for layer 0, then all for layer 1, etc).
      - If 'use_schedule' is True and 'layers' is None, we invoke the C++
        scheduler rmcore.tdepth_schedule_from_monoms on the set of monomials
        with odd coefficients (T-like), and use its layers. Even-coefficient
        monomials (S-like etc) are placed after all scheduled T-layers.

    Fallback:
      - If neither 'layers' is provided nor 'use_schedule' is True, this behaves
        exactly as before (original sequential synthesis order by mk_positions).
    """
    # helpers to emit a single monomial t gate: gather parity -> phase -> uncompute
    def _emit_monomial(circ: Circuit, mask: int, k: int):
        if (k % 8) == 0:
            return
        # Choose target as least significant set bit in mask.
        t = (mask & -mask).bit_length() - 1
        rem = mask & ~(1 << t)

        # forward: gather parity onto t
        b = rem
        while b:
            i = (b & -b).bit_length() - 1
            circ.add_cnot(i, t)
            b &= b - 1

        # phase on the target
        circ.add_phase(t, k % 8)

        # backward: uncompute parity (reverse CNOTs)
        b = rem
        stack: List[int] = []
        while b:
            i = (b & -b).bit_length() - 1
            stack.append(i)
            b &= b - 1
        for i in reversed(stack):
            circ.add_cnot(i, t)

    # build coeffs dict of non-zero entries mod 8
    coeffs = vec_to_coeffs(vec, n)
    if not coeffs:
        return Circuit(n)

    # if needed, compute layers from the C++ scheduler using odd monomials
    if layers is None and use_schedule:
        # robust import of rmcore
        rmcore_mod = None
        try:
            from . import rmcore as _rmcore
            rmcore_mod = _rmcore
        except Exception:
            try:
                import rmcore as _rmcore
                rmcore_mod = _rmcore
            except Exception:
                rmcore_mod = None

        if rmcore_mod is not None:
            odd_monoms = [m for (m, v) in coeffs.items() if (v & 1)]
            if odd_monoms:
                try:
                    _depth, layers = rmcore_mod.tdepth_schedule_from_monoms(odd_monoms, n, schedule_budget)
                except Exception:
                    layers = None  # fall back below
        # if scheduler missing or failed, leave layers = None and fall back

    circ = Circuit(n)

    if layers is None:
        # ORIGINAL behavior: sequential synthesis by mk_positions order.
        order, _ = mk_positions(n)
        for j, k in enumerate(vec):
            k %= 8
            if k == 0:
                continue
            mask = order[j]
            _emit_monomial(circ, mask, k)
        return circ

    # depth-aware: place layer-by-layer

    # 1) emit all odd monomials that appear in the given layers, in order
    scheduled = set()
    for layer in layers:
        for m in layer:
            k = coeffs.get(m, 0) % 8
            if k == 0:
                continue
            _emit_monomial(circ, m, k)
            scheduled.add(m)

    # 2) emit any remaining monomials not in 'layers' (e.g., even coefficients,
    #    or odd monomials the schedule omitted). We emit them afterwards so the
    #    T-layer grouping stays contiguous in the construction.
    order, _ = mk_positions(n)
    for mask in order:
        if mask in scheduled:
            continue
        k = coeffs.get(mask, 0) % 8
        if k == 0:
            continue
        _emit_monomial(circ, mask, k)

    return circ

# Scheduler helpers: compute layers + synthesize in one call

def _try_load_rmcore():
    """Best-effort loader for the native rmcore module (returns None if unavailable)."""
    try:
        from . import rmcore as _rmcore  # relative import to avoid cycle
        return _rmcore
    except Exception:
        pass
    try:
        import rmcore as _rmcore          # local build / flat layout
        return _rmcore
    except Exception:
        return None

def compute_tdepth_layers_from_vec(
        vec: List[int],
        n: int,
        budget: int = -1,
) -> Tuple[Optional[int], Optional[List[List[int]]]]:
    """
    Compute T-depth layers for the odd (T-like) monomials present in 'vec'
    on n qubits, using the native scheduler if available.

    Returns (depth, layers):
      - depth: int if scheduler succeeded, else None
      - layers: List[List[int]] of monomial masks in layer order, or None
        if the scheduler is unavailable.
      - If there are no odd monomials, returns (0, []).
    """
    rmcore_mod = _try_load_rmcore()
    if rmcore_mod is None:
        return None, None

    coeffs = vec_to_coeffs(vec, n)
    odd_monoms = [m for (m, v) in coeffs.items() if (v & 1)]
    if not odd_monoms:
        return 0, []

    depth, layers = rmcore_mod.tdepth_schedule_from_monoms(odd_monoms, n, budget)
    return int(depth), layers

def synthesize_with_schedule(
        vec: List[int],
        n: int,
        budget: int = -1,
) -> Tuple[Circuit, Optional[int], Optional[List[List[int]]]]:
    """
    One-liner convenience:
      1) Compute layers with the native scheduler (if available),
      2) Synthesize using those layers (depth-aware),
      3) Return (circuit, depth, layers).

    If the scheduler is unavailable, falls back to the standard sequential
    synthesis and returns (circ, None, None). If there are no odd monomials,
    returns (circ, 0, []).
    """
    depth, layers = compute_tdepth_layers_from_vec(vec, n, budget)
    if layers is not None:
        circ = synthesize_from_coeffs(vec, n, layers=layers)
    else:
        circ = synthesize_from_coeffs(vec, n)  # fallback, preserves original behavior
    return circ, depth, layers

def t_count_of_coeffs(vec: List[int]) -> int:
    """Count odd coefficients in vec (T-count of the polynomial)."""
    return sum(1 for v in vec if (v & 1) == 1)

def optimize_coefficients(a_coeffs: Dict[int, int], n: int):
    """
    Simple brute-force optimizer using the pure-Python ML decoder.

    Steps:
      1) Convert sparse coeff dict to dense vec.
      2) Compute oddness mask w_bits.
      3) Decode in RM(r,n) via decode_rm_bruteforce (pure Python).
      4) Lift selected monomials to c, add to vec to get vec_opt.
      5) Build a DecodeReport-like object describing the result.

    This is mainly for testing/debug; the production pipeline uses the
    native decoders exposed via rmcore.
    """
    vec = coeffs_to_vec(a_coeffs, n)
    L = len(vec); r = n - 4
    w_bits = res2_vec(vec)
    if r < 0:
        from .report import DecodeReport
        return vec, DecodeReport(n=n, r=r, length=L, dimension=0, distance=w_bits.bit_count(),
                                 selected_monomials=[], codeword_bits=0, ties=1), []
    cw, selected, ties = decode_rm_bruteforce(w_bits, n, r)
    c_vec = lift_to_c(selected, n)
    vec_opt = add_mod8_vec(vec, c_vec)
    from .report import DecodeReport
    rep = DecodeReport(n=n, r=r, length=L, dimension=rm_dimension(n, r), distance=(cw ^ w_bits).bit_count(),
                       selected_monomials=selected, codeword_bits=cw, ties=ties)
    return vec_opt, rep, selected
