# Overview

This library implements and extends the Reed–Muller decoding–based optimization framework for linear‑phase quantum circuits introduced by Amy and Mosca in T‑Count Optimization and Reed–Muller Codes.

At a high level, the workflow is:

1. **Represent a CNOT+phase circuit as a phase polynomial** $U_f \colon |x\rangle \mapsto e^{i\pi f(x)/4} |x\rangle$
where $f(x)$ is a polynomial over $\mathbb{Z}_8$ in Boolean variables.

2. **Collect the coefficients** $a_m \in \mathbb{Z}_8$ of monomials ($x^m = \prod_{i\in S} x_i$) into a length $2^n-1$ vector $a$ where $f(x)$ is a polynomial over $\mathbb{Z}_8$ in Boolean variables.

3. **Take the parity** ($\bmod 2$) of the coefficients to get a binary word $w = \operatorname{Res}_2(a) \in \{0,1\}^{2^n-1}.$

4.  **Interpret $w$ as an element of a punctured Reed–Muller code** $\mathrm{RM}(n-4,n)^*$. Finding the best way to add a codeword $c$ to $w$ (minimum‑distance decoding) $c^\star = \arg\min_{c \in \mathrm{RM}(n-4,n)^*} d_H(w,c)$ corresponds exactly to minimizing the T‑count of the circuit.

5.  **Apply the correction** $c^\star$ back in coefficient space, then re‑synthesize the circuit from the optimized coefficients (optionally using a T‑depth scheduler).


This repository provides:

* Efficient RM decoders in C++ (Dumer, Dumer‑List, Dumer‑List+Chase, RPA‑based), exposed via a `rmcore` Python extension.
* A flexible Python decoding front‑end that combines these decoders with additional heuristics ($\mathrm{GL}(n,2)$ search, OSD, local search).
* A user‑facing Optimizer API that takes a Circuit (CNOT+phase) and returns an optimized circuit plus a detailed report.
* A multi‑order rotation optimizer for $R_Z(2\pi / d)$ gates (generalizing the T‑gate case).
* A T‑depth scheduler based on graph coloring (DSATUR) applied to parity gadgets.
* Autotuning, benchmarking, contracts, and CLI tools.

---

## Glossary

This glossary is useful to get familiar with the terms are used in the codebase:

**Monomial mask / mask**
-   An integer in $1 \dots 2^n-1$ whose binary representation selects which variables appear in a monomial:
`mask & (1 << i)` nonzero $\iff$ monomial includes $x_i$.
Degree of a mask: number of 1‑bits (`popcount(mask)`).

**Phase polynomial / phase coefficients**
-   A function $f : \{0,1\}^n \to \mathbb{Z}_8$ written as

$$
f(x) = \sum_m a_m \cdot x^m
$$

where $x^m$ ranges over all nonzero monomials (and sometimes the constant monomial), and $a_m \in \mathbb{Z}_8$ (or $\mathbb{Z}_{2^k}$, $\mathbb{Z}_d$). In the code:
Stored as a dict `{mask -> coeff}` or as a dense vector `[a_m]`.

**Oddness / odd set / parity vector**
-   For coefficients $a_m$, the oddness is the vector of parity bits:

$$
w_m = a_m \bmod 2.
$$

It is this binary vector that is decoded in the RM code.

**T‑count**
-   The number of monomials with odd coefficient in a Z₈ phase polynomial:

$$
\text{T‑count} = |\{m : a_m \equiv 1 \pmod 2\}|.
$$

In other words, the Hamming weight of the oddness vector.

**T‑depth / T‑layers**
-   Given a set of T‑like phase gates, T‑depth is the minimal number of sequential layers of such gates, assuming arbitrary parallelization of commuting/non‑conflicting T gates. In our parity‑gadget model, two monomials conflict if they share a variable, so T‑layers correspond to color classes of a conflict graph.

**Reed–Muller code $\mathrm{RM}(r,n)$**
-   A linear binary code whose codewords are evaluations of Boolean polynomials on $\{0,1\}^n$ of degree $\le r$. The code has length $2^n$, and we often use its punctured version of length $2^n-1$ by omitting the all‑zero input. (Error Correction Zoo)

**Punctured RM code $\mathrm{RM}(r,n)^*$**
-   The code obtained by removing the coordinate corresponding to the all‑zero input from the full truth table of $\mathrm{RM}(r,n)$ codewords.

**RM(r,n) generator rows**
-   Row $i$ of the generator matrix is the truth table (punctured) of the monomial corresponding to some mask `monoms[i]`. Adding rows corresponds to adding monomials mod 2.

**Minimum‑distance decoding**
-   Given a received word $w$ and a code $C$, find a codeword $c \in C$ minimizing the Hamming distance $d_H(w,c)$.

**Dumer decoding**
-   A recursive decoding algorithm for RM(r,n) codes based on Plotkin’s $(u, u+v)$ decomposition and successive decoding of subcodes. (arXiv)

**RPA (Recursive Projection–Aggregation) decoding**
-   A family of algorithms that recursively project a codeword onto cosets, decode lower‑order RM codes, and aggregate the projections (often by majority). (arXiv)

**OSD (Ordered Statistics Decoding)**
-   A near‑ML decoding method that orders bits by reliability, chooses an information set, solves for coefficients, and enumerates low‑weight error patterns in that information set. (Semantic Scholar)

**GL(n,2) search / preconditioning**
-   Applying random invertible linear maps $A \in \mathrm{GL}(n,2)$ to the coordinate indices to re‑label monomials and potentially make decoding easier, then undoing the map afterwards.

**Bit‑plane / multi‑order optimization**
-   For rotations $R_Z(2\pi / 2^k)$ (or more general $R_Z(2\pi/d)$), we decompose coefficients into $k$ binary planes corresponding to the bits of the exponent. Each plane is treated as a separate binary word to decode in a different RM code, then recombined.