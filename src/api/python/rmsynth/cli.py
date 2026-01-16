"""
rmsynth-optimize: optimize a phase polynomial (Z8 vector) using RM decoders.

Usage examples:
  rmsynth-optimize --decoder rpa --effort 3 --n 6 --gen near1 --flips 10
  rmsynth-optimize --decoder dumer-list --effort 4 --vec-json vec_n6.json --json report.json

This small CLI is mainly a convenience / demo tool: it generates or loads a Z8 vector of length 2^n-1, synthesizes a circuit, runs the Optimizer, and prints before/after T-count and the decoder configuration used.
"""
from __future__ import annotations
import argparse, json, sys
from typing import List
from .optimizer import Optimizer
from .core import synthesize_from_coeffs

def gen_vec(n: int, gen: str, flips: int, density: float, seed: int) -> List[int]:
    """
    Generate a test Z8 vector of length 2^n-1.

    gen == "near1":
        Start from oddness = 1...1 and flip `flips` positions to 0 (random).

    gen == "rand_sparse":
        Oddness is Bernoulli(density), even bits zero.

    gen == "rand_z8":
        Oddness Bernoulli(density); even bits uniform in {0,2,4,6}.
    """
    import random
    rnd = random.Random(seed)
    L = (1 << n) - 1
    if gen == "near1":
        w = [1] * L
        idx = list(range(L)); rnd.shuffle(idx)
        for k in idx[:min(flips, L)]: w[k] = 0
        return [1 if b else 0 for b in w]
    if gen == "rand_sparse":
        return [1 if rnd.random() < density else 0 for _ in range(L)]
    if gen == "rand_z8":
        vec = []
        for _ in range(L):
            odd = 1 if rnd.random() < density else 0
            even = 2 * rnd.choice([0,1,2,3])
            vec.append((even | odd) % 8)
        return vec
    raise SystemExit(f"Unknown generator: {gen}")

def main(argv=None):
    """
    Parse CLI arguments, build or load a vector, optimize it, and print a summary.

    Options:
      --decoder : which decoder strategy to use ('auto' selects heuristically).
      --effort  : coarse effort knob passed to the Optimizer.
      --vec-json: instead of generating, load a JSON list of ints from this file.
      --json    : optional path to dump a JSON summary report.
    """
    p = argparse.ArgumentParser(prog="rmsynth-optimize")
    p.add_argument("--decoder", type=str, default="rpa",
                   choices=["dumer","dumer-list","rpa","ml-exact","auto"])
    p.add_argument("--effort", type=int, default=3)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--vec-json", type=str, default="")
    p.add_argument("--gen", type=str, default="near1", choices=["near1","rand_sparse","rand_z8"])
    p.add_argument("--flips", type=int, default=10)
    p.add_argument("--density", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--json", type=str, default="", help="write report JSON here")
    args = p.parse_args(argv)

    if args.vec_json:
        vec = json.load(open(args.vec_json))
        if not isinstance(vec, list): raise SystemExit("vec-json must be a JSON list of ints")
    else:
        vec = gen_vec(args.n, args.gen, args.flips, args.density, args.seed)

    circ = synthesize_from_coeffs(vec, args.n)
    opt = Optimizer(decoder=args.decoder, effort=args.effort)
    new_circ, rep = opt.optimize(circ)

    print(f"decoder={args.decoder} effort={args.effort} n={rep.n}")
    print(f"T-before={rep.before_t}  T-after={rep.after_t}  distance={rep.distance}")
    print(f"used strategy: {opt.last_decoder_used}")
    if args.json:
        out = {
            "decoder": args.decoder, "effort": args.effort,
            "n": rep.n, "before_t": rep.before_t, "after_t": rep.after_t,
            "distance": rep.distance, "strategy": opt.last_decoder_used,
            "selected_monomials": rep.selected_monomials, "signature": rep.signature,
        }
        json.dump(out, open(args.json,"w"), indent=2)
        print(f"Wrote {args.json}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
