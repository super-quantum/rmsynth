# 13. References

This section summarizes the main theoretical references that underpin the design of the library. The implementation is not a verbatim copy of any one paper, but the architecture and algorithms are closely inspired by this body of work.

## 13.1 Primary reference: T-count optimization via RM codes

The main reference for the overall approach (using Reed–Muller decoding to optimize T-counts of Clifford+T circuits) is:

> **M. Amy and M. Mosca**, “T-count optimization and Reed–Muller codes,” *IEEE Transactions on Information Theory*, 2019. (preprint available as **arXiv:1601.07363**).

This paper establishes the connection between phase polynomials and punctured RM codes, analyzes how decoding corresponds to T-count minimization, and discusses several decoder families. The overall structure of this library, including the use of punctured RM(r,n) with $r = n − 4$, the emphasis on T-count, and the use of Dumer-like and RPA-like decoders, follows that framework.

## 13.2 Reed–Muller decoding

Several classical and modern decoding algorithms for Reed–Muller codes inform the choices made here:

* **I. Dumer**, “Recursive decoding of Reed–Muller codes,” *Allerton 1999*, and follow‑up works.
  These papers introduce and analyze recursive hard-decision decoders for RM codes. The `dumer_decode_full` and its list variants are closely aligned with this style of decoding.

* **M. Ye and E. Abbe**, “Recursive Projection–Aggregation Decoding of Reed–Muller Codes,” *IEEE Transactions on Information Theory*, 2020. (**arXiv**)
  This work introduces the RPA family of decoders, based on projecting onto cosets, recursively decoding, and aggregating with majority votes. The `rpa1_iter` and `rpa1_seed_full` routines implement a tailored RPA‑1 style seeding algorithm adapted to the T-count optimization setting.

* **Automorphism / GL(n,2) search.**
  The use of GL(n,2) transformations to precondition the punctured codeword and search over RM automorphisms fits into the broader class of automorphism-based decoders and preconditioning techniques for RM codes (see, e.g., discussions in Abbe et al.’s survey on RM codes).

## 13.3 Ordered Statistics Decoding (OSD)

The OSD routines in `osd.py` are based on classical ordered-statistics decoding:

> **M. P. C. Fossorier**, “Soft-desicion Decoding of Linear Block Codes Based on Ordered Statistics”.

A variety of later papers refine and extend OSD, but the core pattern remains: choose a reliable information set, compute a base codeword, and explore low-weight error patterns in that information set. The implementation here follows that conceptual blueprint for low orders (L1–L3), adapted to the RM generator and customized with depth-aware tie-breaking via the T-depth estimator.

## 13.4 T-depth optimization and parity networks

The scheduling and T-depth computation features are related to several strands of work on T-depth and parity networks:

> **M. Amy, D. Maslov, and M. Mosca**, “Polynomial-time T-depth optimization of Clifford+T circuits via matroid partitioning,” *IEEE TCAD*, 2014.

This earlier paper focuses on T-depth rather than T-count, and uses matroid partitioning to optimize depth. The library’s T-depth scheduler doesn’t implement that matroid-based approach; instead it models conflicts as a graph of overlapping monomials and uses DSATUR-style coloring (below). The overall goal (grouping T gates into parallel layers under parity network constraints) is shared.

## 13.5 DSATUR graph coloring

The T-depth scheduler uses a DSATUR-style coloring algorithm to greedily and then exactly color the conflict graph of monomials:

> **D. Brélaz**, “New methods to color the vertices of a graph,” *Communications of the ACM*, 22(4):251–256, 1979.

The DSATUR algorithm selects at each step the uncolored vertex with the highest saturation degree (number of different colors used by its neighbors), which tends to produce good colorings in practice. The scheduler in `tdepth.cpp` uses this as the backbone of its branch-and-bound search (with node/time caps).

## 13.6 Additional related work

Finally, a few broader connections worth noting:

* **Meet‑in‑the‑middle circuit synthesis:** many modern compilers and exact-synthesis tools build on meet‑in‑the‑middle and SAT-based search for small subcircuits. The implementation here does not use MITM internally, but it is part of the larger ecosystem of Clifford+T optimization tools described in, e.g., Amy et al.’s “A meet-in-the-middle algorithm for fast synthesis of depth-optimal quantum circuits.”
* **Further RPA and RM literature:** There is an active line of work exploring recursive and projection-based decoders for RM codes and their performance under different channels. The specialized RPA‑1 seeding in this library is deliberately tuned for phase-polynomial optimization rather than channel coding, but conceptually sits in this family.