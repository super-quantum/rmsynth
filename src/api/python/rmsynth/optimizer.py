from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from .core import (
    Circuit, extract_phase_coeffs, synthesize_from_coeffs,
    coeffs_to_vec, t_count_of_coeffs, add_mod8_vec, lift_to_c
)
from .report import OptimizeReport
from .decoders import decode_rm, _load_rmcore
from .autotune import suggest_params  # selector-aware autotuner


# Circuit-level optimizer
#
# This module wraps the decoding layer into a user-facing `Optimizer`  object. The workflow is:
#
#   1. Extract the phase polynomial a(x) from a given Clifford+phase circuit.
#   2. View its odd coefficients (mod 8) as a punctured RM word.
#   3. Call decode_rm(...) using an appropriate strategy (Dumer, RPA, etc.).
#   4. Lift the selected monomials to a correction vector c(x).
#   5. Add c to the original coefficients (mod 8) and re-synthesise a new circuit from the optimised coefficients.
#
# The Optimizer chooses decoder strategies based on `decoder`, `effort`, and optionally an "auto-latency-<Xms>" mode that calibrates RPA-adv via the autotuner (autotune.py).


def _beam_from_effort(effort: int | None) -> int:
    """
    Map a coarse integer 'effort' knob (1..5) to a list_size / beam size.

    effort=None -> default beam 8.
    """
    if effort is None:
        return 8
    effort = max(1, min(int(effort), 5))
    return 1 << effort


def _rpa_iters_from_effort(e: int | None) -> int:
    """
    Heuristic mapping from effort to number of RPA iterations.

    Keeps runtime reasonable while improving quality at higher effort.
    """
    if e is None:
        return 2
    e = max(1, min(int(e), 5))
    return {1: 1, 2: 2, 3: 2, 4: 3, 5: 3}[e]


def _snap_params_from_effort(e: int | None) -> tuple[int, int, bool]:
    """
    Map effort to (snap_t, snap_pool, snap_strong_default).

    This controls how aggressively SNAP explores local neighbourhoods.
    """
    if e is None:
        return (2, 16, False)
    e = max(1, min(int(e), 5))
    if e == 1:
        return (1, 8, False)
    if e == 2:
        return (2, 12, False)
    if e == 3:
        return (2, 16, False)
    if e == 4:
        return (2, 24, True)
    return (3, 24, True)


def _parse_auto_latency(effort: Optional[object]) -> Optional[float]:
    """
    Parse an effort string of the form 'auto-latency-<Xms>'.

    Returns X (float milliseconds) or None if not in auto-latency mode.
    """
    if effort is None or not isinstance(effort, str):
        return None
    s = effort.strip().lower().replace(" ", "")
    if not s.startswith("auto-latency-"):
        return None
    v = s[len("auto-latency-"):]
    if v.endswith("ms"):
        v = v[:-2]
    try:
        return float(v)
    except Exception:
        return None


@dataclass
class CostModel:
    """
    Placeholder for future multi-plane cost models.

    Currently unused except for providing a default mapping for bitplane
    costs (e.g., weighting lower planes differently).
    """
    bitplane_cost: Dict[int, float] = None

    def __post_init__(self):
        if self.bitplane_cost is None:
            self.bitplane_cost = {0: 1.0}


class Optimizer:
    """
    Main user-facing optimisation driver.

    Parameters (common):
      decoder : str or None
          Decoder strategy name ("auto", "rpa", "dumer", etc). If None,
          pick "auto" when rmcore is available and "ml-exact" otherwise.
      effort : int or str or None
          Coarse knob (1..5) controlling beam sizes and RPA iterations, or
          a string of the form "auto-latency-<Xms>" to enable autotuning.
      list_size, chase_t, chase_limit, rpa_iters, osd_top :
          Fine-grained decoding knobs; see decoders.decode_rm.
      snap_* :
          SNAP refinement controls (strength, time/node caps).
      policy, policy_lambda, depth_tradeoff :
          Depth–distance trade-off policy; passed to decode_rm.

    Autotuning:
      When effort is of the form "auto-latency-<Xms>", the optimiser
      chooses RPA-adv parameters for each (n, pre_T) bucket by calling
      autotune.suggest_params, so the expected median decode time is
      below Xms on synthetic workloads.
    """

    def __init__(self,
                 decoder: str | None = None,
                 effort: int | str | None = None,      # int or 'auto-latency-<Xms>'
                 list_size: int | None = None,
                 chase_t: int = 1,
                 chase_limit: int = 16,
                 snap_effort: int | None = None,
                 snap_t: int | None = None,
                 snap_pool: int | None = None,
                 snap_strong: bool | None = None,
                 snap_time_ms: int | None = None,
                 snap_node_limit: int | None = None,
                 rpa_iters: int = 2,
                 osd_top: int | None = None,
                 # linear-cost alias:
                 policy: Optional[str] = None,
                 policy_lambda: Optional[int] = None,
                 # legacy explicit knob (still supported):
                 depth_tradeoff: int | None = None,
                 # control autotuner selector (so calibration & usage match)
                 autotune_selector: Optional[str] = None,     # "quality-under-target" | "pareto"
                 autotune_pareto_dist: Optional[int] = None,  # e.g., 1
                 autotune_pareto_lat: Optional[float] = None, # e.g., 1.10
                 check_contracts: bool = False):
        if decoder is None:
            decoder = "auto" if _load_rmcore() is not None else "ml-exact"
        self.decoder = decoder
        self.effort = effort
        self.list_size = list_size
        self.chase_t = chase_t
        self.chase_limit = chase_limit
        self.snap_effort = snap_effort
        self.snap_t = snap_t
        self.snap_pool = snap_pool
        self.snap_strong = snap_strong
        self.snap_time_ms = snap_time_ms
        self.snap_node_limit = snap_node_limit
        self.rpa_iters = rpa_iters
        self.osd_top = osd_top
        self.policy = policy
        self.policy_lambda = policy_lambda
        self.depth_tradeoff = depth_tradeoff
        self.autotune_selector = autotune_selector
        self.autotune_pareto_dist = autotune_pareto_dist
        self.autotune_pareto_lat = autotune_pareto_lat
        self.check_contracts = check_contracts
        # introspection aids (set on each optimize() call)
        self.last_decoder_used: str | None = None
        self.last_params_used: dict | None = None

    def _choose(self, n: int, before_t: int) -> tuple[str, dict]:
        """
        Select a decoder strategy + kwargs for a given instance.

        Preference order:
          - if effort is 'auto-latency-<Xms>' and rmcore is present, use
            autotuned RPA-adv parameters for the requested latency target;
          - otherwise honour explicit decoder choices;
          - if decoder == "auto", pick a heuristic strategy based on n
            and the initial T-count before_t (small -> Dumer, medium ->
            Dumer-list, large/high-T -> RPA-adv).
        """

        policy_kwargs = {}
        if self.policy is not None or self.depth_tradeoff is not None:
            policy_kwargs["policy"] = self.policy
            policy_kwargs["policy_lambda"] = self.policy_lambda
            policy_kwargs["depth_tradeoff"] = self.depth_tradeoff

        # auto-latency path: use autotuner + rpa-adv
        auto_ms = _parse_auto_latency(self.effort)
        if auto_ms is not None and _load_rmcore() is not None:
            params = suggest_params(
                n, before_t, auto_ms,
                policy=self.policy, policy_lambda=self.policy_lambda,
                selector=self.autotune_selector,
                pareto_dist_slack=self.autotune_pareto_dist,
                pareto_latency_factor=self.autotune_pareto_lat
            )
            kwargs = {
                "list_size": params.beam,
                "rpa_iters": params.rpa_iters,
                "snap": True,
                "snap_t": params.snap_t,
                "snap_pool": params.snap_pool,
                "snap_strong": params.snap_strong,
                **policy_kwargs
            }
            return "rpa-adv", kwargs

        # explicit decoders (unchanged)
        if self.decoder == "ml-exact":
            return "ml-exact", {}
        if self.decoder == "dumer":
            return "dumer", {}
        if self.decoder == "dumer-list":
            ls = self.list_size or _beam_from_effort(self.effort if isinstance(self.effort, int) else None)
            return "dumer-list", {"list_size": ls, **policy_kwargs}
        if self.decoder == "dumer-list-adv":
            ls = self.list_size or _beam_from_effort(self.effort if isinstance(self.effort, int) else None)
            ct = self.chase_t if self.chase_t is not None else 2
            cl = self.chase_limit if self.chase_limit is not None else 16
            return "dumer-list-chase", {"list_size": ls, "chase_t": ct, "chase_limit": cl, **policy_kwargs}
        if self.decoder in ("osd1", "osd2", "osd3"):
            ls = self.list_size or _beam_from_effort(self.effort if isinstance(self.effort, int) else None)
            return self.decoder, {"list_size": ls, **policy_kwargs}
        if self.decoder in ("beam-osd1", "beam-osd2", "beam-osd3"):
            ls = self.list_size or _beam_from_effort(self.effort if isinstance(self.effort, int) else None)
            top = self.osd_top or min(ls, 8)
            return self.decoder, {"list_size": ls, "osd_top": top, **policy_kwargs}
        if self.decoder == "rpa":
            # RPA-1 + advanced SNAP / OSD
            ls = self.list_size or _beam_from_effort(self.effort if isinstance(self.effort, int) else None)
            it = self.rpa_iters if self.rpa_iters is not None else _rpa_iters_from_effort(
                self.effort if isinstance(self.effort, int) else None
            )
            st, sp, strong_default = _snap_params_from_effort(
                self.snap_effort if self.snap_effort is not None else (self.effort if isinstance(self.effort, int) else None)
            )
            st = self.snap_t or st
            sp = self.snap_pool or sp
            strong = self.snap_strong if self.snap_strong is not None else strong_default
            kwargs = {
                "list_size": ls,
                "rpa_iters": it,
                "snap": True,
                "snap_t": st,
                "snap_pool": sp,
                "snap_strong": strong,
                **policy_kwargs,
            }
            if self.snap_time_ms is not None:
                kwargs["snap_time_ms"] = self.snap_time_ms
            if self.snap_node_limit is not None:
                kwargs["snap_node_limit"] = self.snap_node_limit
            return "rpa-adv", kwargs
        if self.decoder == "rpa2":
            ls = self.list_size or _beam_from_effort(self.effort if isinstance(self.effort, int) else None)
            it = self.rpa_iters if self.rpa_iters is not None else _rpa_iters_from_effort(
                self.effort if isinstance(self.effort, int) else None
            )
            st, sp, strong_default = _snap_params_from_effort(
                self.snap_effort if self.snap_effort is not None else (self.effort if isinstance(self.effort, int) else None)
            )
            st = self.snap_t or st
            sp = self.snap_pool or sp
            strong = self.snap_strong if self.snap_strong is not None else strong_default
            kwargs = {
                "list_size": ls,
                "rpa_iters": it,
                "snap": True,
                "snap_t": st,
                "snap_pool": sp,
                "snap_strong": strong,
                **policy_kwargs,
            }
            if self.snap_time_ms is not None:
                kwargs["snap_time_ms"] = self.snap_time_ms
            if self.snap_node_limit is not None:
                kwargs["snap_node_limit"] = self.snap_node_limit
            return "rpa2", kwargs

        if self.decoder == "auto":
            # heuristic defaults depending on problem size
            if _load_rmcore() is None:
                return "ml-exact", {}

            # high-complexity / larger circuits: use RPA
            if n >= 7 or before_t >= 24:
                ls = self.list_size or _beam_from_effort(self.effort if isinstance(self.effort, int) else None)
                it = self.rpa_iters if self.rpa_iters is not None else _rpa_iters_from_effort(
                    self.effort if isinstance(self.effort, int) else None
                )
                st, sp, strong_default = _snap_params_from_effort(
                    self.snap_effort if self.snap_effort is not None else (self.effort if isinstance(self.effort, int) else None)
                )
                st = self.snap_t or st
                sp = self.snap_pool or sp
                strong = self.snap_strong if self.snap_strong is not None else strong_default
                kwargs = {
                    "list_size": ls,
                    "rpa_iters": it,
                    "snap": True,
                    "snap_t": st,
                    "snap_pool": sp,
                    "snap_strong": strong,
                    **policy_kwargs,
                }
                if self.snap_time_ms is not None:
                    kwargs["snap_time_ms"] = self.snap_time_ms
                if self.snap_node_limit is not None:
                    kwargs["snap_node_limit"] = self.snap_node_limit
                return "rpa-adv", kwargs

            # medium size: prefer plain Dumer-list (NO chase)
            if n >= 6 or before_t >= 16:
                ls = self.list_size or _beam_from_effort(self.effort if isinstance(self.effort, int) else None)
                return "dumer-list", {"list_size": ls, **policy_kwargs}

            # very small: plain Dumer
            return "dumer", {}

        # fallback: honour the string verbatim (for advanced / experimental strategies)
        return self.decoder, {}

    def optimize(self, circ: Circuit) -> Tuple[Circuit, OptimizeReport]:
        """
        Optimise the phase polynomial of `circ` and return a new circuit.

        Steps:
          1. Extract phase coefficients (mod 8) from the input circuit.
          2. Compute its T-count (number of odd coefficients).
          3. Choose a decoder and parameters via _choose().
          4. Decode the punctured oddness word to an RM codeword.
          5. Lift selected monomials to a correction vector c(x).
          6. Add c(x) to the original coefficients and re-synthesise.

        Returns (new_circuit, OptimizeReport).
        """
        n = circ.n
        a = extract_phase_coeffs(circ)
        vec = coeffs_to_vec(a, n)
        before_t = t_count_of_coeffs(vec)

        r = n - 4
        L = len(vec)
        # oddness of phase polynomial (punctured)
        w_bits = [1 if (v & 1) else 0 for v in vec]

        strategy, kwargs = self._choose(n, before_t)
        self.last_decoder_used = strategy
        self.last_params_used = kwargs

        if r < 0:
            # degenerate case: RM(r<0,n) is {0}; we cannot improve
            code_bits = [0] * ((1 << n) - 1)
            selected = []
            dist = sum(w_bits)
        else:
            code_bits, selected, dist = decode_rm(w_bits, n, r, strategy, **kwargs)

        # lift correction and apply to coefficients
        c_vec = lift_to_c(selected, n)
        vec_opt = add_mod8_vec(vec, c_vec)

        after_t = t_count_of_coeffs(vec_opt)
        new_circ = synthesize_from_coeffs(vec_opt, n)

        # optional consistency checks (contracts.py)
        if self.check_contracts:
            from .contracts import check_all, maybe_check_all
            check_all(vec, n, r, code_bits, selected, dist, strict=True)
        else:
            from .contracts import maybe_check_all
            maybe_check_all(vec, n, r, code_bits, selected, dist)

        import hashlib
        sig = hashlib.sha256(bytes(vec_opt)).hexdigest()
        rep = OptimizeReport(
            n=n,
            before_t=before_t,
            after_t=after_t,
            distance=dist,
            r=r,
            bitlen=L,
            selected_monomials=selected,
            signature=sig,
        )
        return new_circ, rep
