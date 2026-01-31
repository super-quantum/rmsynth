# 4. The C++ core (rmcore)

The native core is where all the heavy bit‑level work happens: RM generator matrices, decoders, ANF, and the T‑depth scheduler. Everything is written in terms of compact bit vectors and exposed into Python via a small set of `pybind11` bindings.

The following sections walk through the main pieces and how they fit together.

## 4.1 Bit operations and representations (`bitops.h`)

Almost every C++ file in the core sits on top of the `BitVec` struct defined in `bitops.h`:

```cpp
struct BitVec {
    int nbits;
    std::vector<uint64_t> w;

    BitVec(): nbits(0) {}
    explicit BitVec(int n): nbits(n), w((n+63)>>6, 0ull) {}
    int size() const { return nbits; }

    void clear(){ std::fill(w.begin(), w.end(), 0ull); }
    // ...
};
```

The representation is:

* `nbits`: the logical length in bits;
* `w`: the underlying storage as a vector of 64‑bit words.

Bits are stored in little‑endian fashion within each word: bit `i` lives at `w[i >> 6]` bit `(i & 63)`.



Convenience methods provide the usual bit operations:

* `get(i)`, `set1(i)`, `set0(i)`, and `toggle(i)` read and write individual bits;
* `xor_inplace(other)` XORs another `BitVec` of the same length into this one, word by word;
* `slice(start, len)` returns a new `BitVec` with a contiguous slice of bits;
* `concat(a, b)` returns a new `BitVec` formed by appending `b` after `a`;
* `clear()` resets all bits to zero.

Two operations are performance‑critical and therefore written with care:

1.  `weight()` computes the Hamming weight of the vector. The implementation loops over the 64‑bit words, and for each word uses `__builtin_popcountll` where available, falling back to a small loop that clears the lowest set bit repeatedly. When OpenMP is enabled, the outer loop is annotated with `#pragma omp simd reduction(+:s)` to allow vectorization and reduction.
2.  `hamming_distance(a, b)` wraps `hamming_distance_words(a.w.data(), b.w.data(), a.w.size())`, which XORs words and popcounts each, with the same OpenMP‑friendly structure.

Everything else in the core (Dumer, list decoding, RPA, T‑depth estimation) works at the level of `BitVec` and these helpers, so it’s worth keeping the representation and semantics in mind.

## 4.2 RM generator and brute‑force ML decoding (C++)

The C++ implementation of RM generator matrices and dimensions mirrors the Python reference but returns more low‑level structures.
In `rm_code.cpp`:

```cpp
std::vector<uint64_t> rm_generator_rows(int n, int r, std::vector<int>& monoms);
int rm_dimension(int n, int r);
```

`rm_generator_rows` builds the generator rows for $\mathrm{RM}(r,n)^*$ in punctured form (length $2^n-1$).
It fills `monoms` with the list of monomial masks (starting with 0 for the constant term if $r \ge 0$), and returns a vector of 64‑bit words where each entry is a row. For a given row and column index $j$, bit $j$ indicates whether the corresponding monomial evaluates to 1 on the nonzero input $j+1$. The heart of the construction is:

```cpp
for(int t=1; t<(1<<n); ++t){
    if (weight(t) <= r){
        monoms.push_back(t);
        uint64_t bits=0;
        for(int y=1; y<=L; ++y)
            if ((t & y) == t) bits |= (1ull<<(y-1));
        rows.push_back(bits);
    }
}
```

`rm_dimension` computes the dimension $\sum_{d=0}^r \binom{n}{d}$ using an incremental binomial recurrence.

The C++ brute‑force decoder lives in `decode_bruteforce.cpp`:

```cpp
struct DecodeResult{
    uint64_t codeword;
    std::vector<int> monomials;
    int ties;
};

DecodeResult decode_rm_bruteforce(uint64_t w_bits, int n, int r);
```

This is the native analogue of the Python reference decoder:

1.  It calls `rm_generator_rows(n, r, monoms)` to get the generator matrix rows and monomial list.
2.  It iterates over all coefficient vectors $u$ in $\mathrm{GF}(2)$ of length `m = rows.size()`; for each $u$ it forms a codeword by XORing the selected rows:
    ```cpp
    uint64_t cw=0;
    for (int i=0;i<m;++i) if ((u>>i)&1) cw ^= rows[i];
    ```
3.  It computes the Hamming distance to the received word `w_bits` using `popcount64`.
4.  It keeps track of the best distance, the corresponding codeword, and the coefficient vector, breaking ties lexicographically on `cw` and counting how many ML ties were encountered.

The result is bundled into `DecodeResult`. This function is intended for testing and very small $n$ only; in production everything routes through Dumer, list decoding, or RPA.

## 4.3 Dumer decoders

### `dumer_decode_full` — recursive Dumer decoder

The baseline Dumer algorithm is implemented in `dumer.cpp` as:

```cpp
BitVec dumer_decode_full(const BitVec& y_full, int r, int n);
```

Here `y_full` is a full length $2^n$ truth table for a received word over $\mathbb{F}_2$; `r` and `n` define the $\mathrm{RM}(r,n)$ code.



The function uses the Plotkin decomposition:

* **base cases:**
    * if `r < 0`, the only codeword is all zeros;
    * if `r >= n`, the code is the whole space, so it returns `y` itself;
    * if `n == 0`, there is only a single coordinate;
    * if `r == 0`, the code contains only the all‑zeros and all‑ones words, and the decoder picks whichever is closer to `y` by Hamming weight.
* **recursive case for $0 < r < n$:**
    1.  split `y` into halves `y0` and `y1` of length $2^{n-1}$;
    2.  form their sum `ysum = y0 ^ y1`;
    3.  decode `v_hat = dumer_decode_full(ysum, r-1, n-1)` in $\mathrm{RM}(r-1, n-1)$;
    4.  compute `tmp = y0 ^ v_hat`;
    5.  decode `u_hat = dumer_decode_full(tmp, r, n-1)` in $\mathrm{RM}(r, n-1)$;
    6.  reconstruct the full codeword as `(u_hat, u_hat ^ v_hat)`.

All of this is expressed directly in terms of `BitVec` slices, XORs, and concatenations.

### `dumer_list_decode_full` — list decoder

`dumer_list.cpp` extends the above to list decoding:

```cpp
struct DL_Candidate {
    BitVec code;
    int dist;
};

std::vector<DL_Candidate>
dumer_list_decode_full(const BitVec& y, int r, int n, int list_size);
```

Instead of returning a single codeword, this version returns up to `list_size` candidates with the smallest Hamming distance to `y`. The recursion mirrors `dumer_decode_full`, but at each step it:

1.  Recursively decodes all possibilities for $v$ into a list `v_list`.
2.  For each `vCand` in `v_list`, it constructs an input for $u$ as `tmp = y0 ^ vCand.code`, then recursively decodes `u_list` for that `tmp`.
3.  For each pair (`uCand`, `vCand`) it builds the full codeword and computes its distance to `y`.

These `(codeword, dist)` pairs are accumulated into a big vector `out`. At the end of the recursion, `trim(out, list_size)` is called to keep only the best candidates.

The `trim` function is a key piece:

* In the fast, non‑deterministic path (default), it uses `std::nth_element` to partition the vector by distance, truncates it to keep entries, and then `std::stable_sort`s that truncated range by distance.
* If the environment variable `RM_DETERMINISTIC` is set to a “truthy” value, the deterministic path is used instead. It computes a key for each candidate:
    ```cpp
    std::tuple<int, uint64_t, std::size_t>(dist, hash_bitvec(code), index)
    ```
  where `hash_bitvec` is a 64‑bit FNV‑1a hash of the bitvector contents and `nbits`. These keys are stably sorted, and the first keep candidates are selected in that order. This guarantees a reproducible ordering for tests and reproducible results across runs when needed.

The recursion can run in parallel using the abstraction in `parallel.h`. When `RM_HAVE_TBB` or `RM_HAVE_OPENMP` is defined and enough candidates are present, the outer loop over `v_list` is parallelized, and each worker builds local candidate vectors which are merged at the end.

### `dumer_list_chase_full` — Chase‑style reliability flips

The file `dumer_list_chase.cpp` builds a Chase‑type decoder on top of `dumer_list_decode_full`:

```cpp
std::vector<DL_Candidate>
dumer_list_chase_full(const BitVec& y, int r, int n,
                      int list_size, int chase_t, int chase_limit);
```

The algorithm is:

1.  Run the baseline list decoder:
    ```cpp
    auto base = dumer_list_decode_full(y, r, n, list_size);
    ```
    If `chase_t <= 0`, `chase_limit <= 0`, or the base list is empty, return base directly.
2.  Take the best baseline candidate `best0 = base[0]` and compute the positions where it disagrees with the observation `y`. These positions are treated as the least reliable bits:
    ```cpp
    for (int i=0;i<y.nbits;i++){
        if (y.get(i) != best0.code.get(i)) errs.push_back(i);
    }
    ```
    Restrict attention to the first `K = min(errs.size(), chase_limit)` error positions.
3.  Enumerate all size‑1 flip patterns, up to `chase_limit` patterns in total: for each index `a` in these positions, flip that bit in `y` to get `yf`, decode a single candidate with `dumer_list_decode_full(yf, r, n, 1)`, and add the resulting candidate to the pool.
4.  If `chase_t >= 2`, also enumerate size‑2 flip patterns `(a, b)`, again capped by `chase_limit`.
5.  Concatenate the baseline candidates and all newly obtained candidates into a single vector, then deduplicate by codeword (using bytewise equality on `BitVec::to_bytes_le()` results).
6.  Finally, call `trim(uniq, list_size)` to keep the best candidates by distance (and hash‑based tie‑breaking in deterministic mode).

The result is a list of candidates that combines Dumer‑List’s breadth with a shallow but targeted local search around the most likely error positions.

## 4.4 RPA‑1 seeding (`rpa.cpp`)

The RPA‑1 machinery is implemented in `rpa.cpp` and provides a way to “polish” the input before applying a standard decoder.



Two static helpers prepare the ground:

* `insert_zero_bit(t, j)` takes a mask `t` over $n-1$ bits and inserts a zero bit at position `j`, embedding it into an $n$-bit space. This is used to map indices when projecting along axis `j`.
* `project_axis_j(y, n, j, y0, y1)` constructs two length $2^{n-1}$ words `y0` and `y1` corresponding to the slices of the full word `y` with $x_j = 0$ and $x_j = 1$, respectively. It uses `insert_zero_bit` to find the base address for each projected index.

The core of RPA‑1 is `rpa1_iter`:

```cpp
static BitVec rpa1_iter(const BitVec& y, int r, int n);
```

For each axis $j = 0 \dots n-1$, it:

1.  Projects the full word into `y0` and `y1`.
2.  Forms `ysum = y0 ^ y1`.
3.  Decodes `v_hat` using `dumer_decode_full(ysum, r-1, n-1)`.
4.  Forms `tmp = y0 ^ v_hat`.
5.  Decodes `u_hat` using `dumer_decode_full(tmp, r, n-1)`.
6.  Recombines them into a full candidate codeword `cand = compose_axis_j(uhat, vhat, n, j)`.

It then aggregates these axis‑specific candidates by majority vote over coordinates. A temporary counter `votes[i]` records how many axes propose a 1 at position `i`. At the end:

* if `votes[i] > n/2`, the next estimate `y_next` has bit `i = 1`;
* if `votes[i] < n/2`, bit `i = 0`;
* if `votes[i] == n/2`, it keeps the original bit `y.get(i)`.

The implementation is written to be parallel‑friendly: the loop over axes is parallelized via `rm_par_for` if TBB or OpenMP is available, and each worker accumulates its own votes array which is reduced at the end. Within each axis, set bits in `cand` are enumerated via bit‑scan (`__builtin_ctzll` or a portable fallback).

On top of this iteration sits `rpa1_seed_full`:

```cpp
BitVec rpa1_seed_full(const BitVec& y, int r, int n, int iters, int final_list_size);
```

It:

1.  Checks that `y.nbits == 1 << n`.
2.  Sets `est = y` and applies `I = max(1, iters)` iterations of `rpa1_iter`, updating `est` each time.
3.  Defines a helper `dist_to_y(code)` that computes the distance to the original observation `y`.
4.  If `final_list_size == 1`, it decodes both:
    * `c_est = dumer_decode_full(est, r, n)`
    * `c_y   = dumer_decode_full(y,   r, n)`
      and returns whichever is closer to `y` in Hamming distance.
5.  If `final_list_size > 1`, it runs Dumer‑List both around `est` and around `y`:
    ```cpp
    auto beam_est = dumer_list_decode_full(est, r, n, ls);
    auto beam_y   = dumer_list_decode_full(y,   r, n, ls);
    ```
    It then scans both beams with `dist_to_y` and picks the overall best candidate. If, for some reason, no candidate is found (which should not happen in practice), it falls back to running `dumer_decode_full(est, r, n)`.

This seeded decoder is exposed to Python both in full‑length form (`decode_rm_rpa1_full`) and, more importantly, wrapped into punctured decoders that compare it against plain list decoding and further refinements in `bindings.cpp` and `decoders.py`.

## 4.5 Multi‑use utilities

Several small headers provide reusable functionality across decoders and wrappers:

### ANF computation (`anf.cpp` / `anf.h`)

```cpp
std::vector<uint8_t> anf_from_truth(const BitVec& y, int n);
```

This function takes a full truth table `y` of length $2^n$ and computes the coefficients of its algebraic normal form via an in‑place Möbius transform.

The implementation:
1.  Copies `y` into a flat vector `a` of bytes, with `a[i]` initialized to `y.get(i) ? 1 : 0`.
2.  For each dimension `d` from 0 to `n-1`, and each mask `mask`, if `mask` has bit `d` set, it XORs `a[mask]` with `a[mask ^ (1<<d)]`.

After this transform, `a[mask]` is the coefficient of the monomial indexed by `mask` (mod 2). In the Python bindings this is used to recover the list of monomials from a full RM codeword.

### Puncturing helpers (`puncture.h`)

Two inline functions manage the transition between full and punctured words:

```cpp
inline BitVec embed_punctured_to_full(const BitVec& punct, bool b0);
inline BitVec puncture_full(const BitVec& full);
```

`embed_punctured_to_full` creates a new `BitVec` of size `punct.nbits + 1`, copies the punctured bits into positions 1.., and sets position 0 to `b0`. `puncture_full` does the inverse: dropping bit 0 and shifting the rest down by one.

Higher‑level decoders (Dumer, list, RPA) are written for full words over $2^n$ coordinates, so the punctured wrappers use these helpers to try both possibilities for the missing bit.

### Parallel abstraction (`parallel.h`)

The function template:

```cpp
template <typename F>
inline void rm_par_for(std::size_t begin, std::size_t end, F f);
```

provides a single for‑loop abstraction which internally selects between:
* a sequential loop;
* a TBB `parallel_for`; or
* an OpenMP `#pragma omp parallel for` loop.

The choice depends on compile‑time macros (`RM_HAVE_TBB`, `RM_HAVE_OPENMP`) and the `RM_PARALLEL` environment variable. If `RM_PARALLEL` is set to a falsy value ("0", "false", "no"…), parallelism is disabled even if TBB or OpenMP are compiled in.

### Miscellaneous (`util.h`)

`util.h` contains a single helper:

```cpp
inline int popcount64(uint64_t x){
#if defined(__GNUG__)
    return __builtin_popcountll(x);
#else
    int c=0; while(x){ x&=(x-1); ++c;} return c;
#endif
}
```

This is used by the C++ brute‑force decoder and in a few other places where a raw 64‑bit Hamming weight is needed without going through `BitVec`.

## 4.6 T‑depth scheduler (`tdepth.cpp` / `tdepth.h`)

The T‑depth scheduling logic is implemented in `tdepth.cpp` and exposed through `tdepth.h`.
There are two main entry points:

```cpp
int estimate_tdepth_from_punctured(const BitVec& odd_punct, int n);
int schedule_tdepth_from_monoms(const std::vector<int>& monoms,
                                int n,
                                int budget,
                                std::vector<std::vector<int>>& layers);
```

The estimator, `estimate_tdepth_from_punctured`, currently returns a very simple upper bound: the Hamming weight of the odd set. This is cheap and safe, and is used during decoding when you just need a rough depth proxy for tie‑breaking.

The scheduler, `schedule_tdepth_from_monoms`, attempts to compute a near‑optimal (often optimal) coloring of the conflict graph built from monomials:



1.  First, `filter_monoms` sorts and deduplicates the input list, then removes the constant monomial (mask 0), since it doesn’t contribute to T‑depth.
2.  Next, `build_graph_from_filtered` constructs a `Graph`:
    ```cpp
    struct Graph {
        int N = 0;
        std::vector<uint64_t> adjBits;   // adjacency bitset (assume N <= 64 for practical sizes)
        std::vector<int> degree;
    };
    ```
3.  Two vertices (monomials) are connected if their supports overlap:
    ```cpp
    if ((monoms[i] & monoms[j]) != 0) {
        G.adjBits[i] |= (1ull << j);
        G.adjBits[j] |= (1ull << i);
        G.degree[i]++; G.degree[j]++;
    }
    ```
    The resulting graph encodes the constraint “two monomials cannot be in the same T layer if they share a variable”.

The actual coloring is done by a DSATUR‑style branch‑and‑bound solver, encapsulated in the `DSAT` struct:

It maintains:
* `color[v]` — the current color assignment of each vertex (-1 for uncolored);
* `satMask[v]` — a bitmask of colors used by colored neighbors of `v`;
* `order_deg[v]` — degrees used for tie‑breaking;
* `best` and `bestColoring` — the best solution found so far.

A greedy coloring `greedy_bound` computes an initial upper bound and a starting coloring, by coloring vertices in order of decreasing degree with the smallest available color.

The depth‑first search `dfs(usedColors)`:
* stops if a node count limit or time limit is reached, or if it can’t beat the current best;
* picks the next vertex to color as the one with maximum saturation degree (number of different colors seen in its neighborhood), tie‑breaking by degree;
* tries all existing colors that are not forbidden, then optionally introduces a new color.

Node and time limits are controlled via environment variables:

* `RM_TDEPTH_COLOR_NODES` (default 500000) — maximum number of search nodes explored;
* `RM_TDEPTH_COLOR_MS` (default 50) — maximum wall‑clock time in milliseconds.

If the search terminates early due to these limits, the solver returns the best coloring it has found so far, which is always at most as bad as the greedy solution; in that case, the scheduler falls back to the greedy coloring state.

`schedule_tdepth_from_monoms` wraps all this and returns:

* `depth` — the minimum number of colors found;
* `layers` — a vector of depth layers, each a vector of monomial masks that can be executed in parallel.

This is the function called from Python (via `rmcore`) when you ask for exact or improved T‑depth scheduling.

## 4.7 Python bindings (`bindings.cpp`)

The last piece of the native core is `bindings.cpp`, which uses `pybind11` to expose a clean, Python‑friendly interface.
Two small helpers convert between `py::bytes` and `BitVec`:

```cpp
static BitVec bytes_to_bitvec(py::bytes b, int nbits) {
    std::string s = b;
    std::vector<uint8_t> v(s.begin(), s.end());
    return BitVec::from_bytes_le(v, nbits);
}

static py::bytes bitvec_to_bytes(const BitVec& bv) {
    auto v = bv.to_bytes_le();
    return py::bytes(reinterpret_cast<const char*>(v.data()), v.size());
}
```

A stable 64‑bit FNV‑1a `hash_bitvec` mirrors the deterministic list‑decoding logic and is used for tie‑breaking in Python wrappers:

```cpp
static inline uint64_t hash_bitvec(const BitVec& b){
    uint64_t h = 1469598103934665603ull;
    for (std::size_t i=0;i<b.w.size();++i){
        uint64_t x = b.w[i];
        for (int k=0;k<8;++k){
            uint8_t byte = (uint8_t)((x >> (8*k)) & 0xffu);
            h ^= byte; h *= 1099511628211ull;
        }
    }
    h ^= (uint64_t)b.nbits;
    return h;
}
```

The module definition `PYBIND11_MODULE(rmcore, m)` then exports:

### RM utilities

```cpp
m.def("rm_dimension", &rm_dimension, py::arg("n"), py::arg("r"));
m.def("rm_generator_rows", &rm_generator_rows,
      py::arg("n"), py::arg("r"), py::arg("monoms"));
```

These return, respectively, the dimension, and the generator rows plus monomial list for $\mathrm{RM}(r,n)^*$.

### Brute‑force decoder

The `DecodeResult` struct from `decode_bruteforce.h` is exposed as a Python class with read‑only `codeword`, `monomials` and `ties` fields, together with:

```cpp
m.def("decode_rm_bruteforce", &decode_rm_bruteforce,
      py::arg("w_bits"), py::arg("n"), py::arg("r"));
```

### Full‑length decoders

```cpp
m.def("decode_rm_dumer_full",
      [](py::bytes y_full_bytes, int n, int r){ ... });

m.def("decode_rm_rpa1_full",
      [](py::bytes y_full_bytes, int n, int r, int iters, int final_list_size){ ... });
```

These are mostly convenience wrappers useful in tests and internal experimentation; the main user‑facing path uses punctured decoders.

### Punctured decoders

The most important bindings are the punctured decoders:

```cpp
m.def("decode_rm_dumer_punctured",      ...);
m.def("decode_rm_dumer_list_punctured", ...);
m.def("decode_rm_dumer_list_topk_punctured", ...);
m.def("decode_rm_dumer_list_chase_punctured", ...);
m.def("decode_rm_rpa1_punctured",       ...);
```

Each of these assumes the input is a punctured word of length $L = 2^n - 1$ encoded as bytes, embeds it into a full word by trying both $b0 = 0$ and $b0 = 1$, decodes using the corresponding full‑length routine (`dumer_decode_full`, `dumer_list_decode_full`, `dumer_list_chase_full`, `rpa1_seed_full`), punctures the result back, and scores candidates by:

1.  punctured Hamming distance to the input;
2.  estimated T‑depth, via `estimate_tdepth_from_punctured` on the residual;
3.  and the hash of the punctured codeword, as a final tie‑breaker.

The RPA‑based bindings (`decode_rm_dumer_list_punctured`, `decode_rm_dumer_list_chase_punctured`, `decode_rm_rpa1_punctured`) additionally convert the chosen full codeword into a list of monomials by computing ANF via `anf_from_truth` and collecting every mask with coefficient 1.

`decode_rm_dumer_list_topk_punctured` is slightly different: it returns only the top_k punctured codewords as Python bytes in a stable sorted order (distance, depth, hash), and leaves monomial recovery and further tie‑breaking to Python.

### T‑depth helpers

Finally, two T‑depth utilities are exported:

```cpp
m.def("tdepth_from_punctured",         ...);
m.def("tdepth_schedule_from_monoms",   ...);
```

`tdepth_from_punctured` takes a punctured odd set as bytes and returns the estimator from `estimate_tdepth_from_punctured`. `tdepth_schedule_from_monoms` takes a list of monomial masks and a budget, calls the DSATUR scheduler, and returns `(depth, layers)` as a Python tuple.

All the bytes↔BitVec conversions, ANF recovery, depth scoring and tie‑breaking remain inside the native module, so that the Python layer can work with plain `List[int]` and `bytes` without worrying about memory layout.