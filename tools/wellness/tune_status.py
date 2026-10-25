#!/usr/bin/env python3
"""Read-only status and assessment of Optuna studies.

usage: tune_status.py <storage_dir> <study> [<study> ...]

Report: trial states, best 5 with parameters, trial durations, running
trials with their epoch progress, remaining time.
Assessment (lines starting with "!"):
  - share of pruned/failed trials, failed trials with divergence epoch
  - per trial: val score rising at the end (last > 1.2 x its minimum after
    >= 3 epochs) -> instability the last-epoch objective will punish
  - per trial: KL / gradient-norm maxima jumping (last epoch > 10 x median)
  - best 5: parameters at the edge of the search space -> extend the range
  - best 5: relative spread of the values (small spread = flat optimum,
    seed noise may dominate the ranking)
  - MTS-JEPA: effective rank of the target embeddings << embed_dim
    (partial representation collapse)
"""
import math, statistics, sys
from datetime import datetime
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
now = datetime.now()


def at_edge(name, value, dist):
    if isinstance(dist, optuna.distributions.CategoricalDistribution):
        ch = list(dist.choices)
        if all(isinstance(c, (int, float)) for c in ch) and len(ch) > 2:
            return value in (min(ch), max(ch))
        return False
    lo, hi = dist.low, dist.high
    if getattr(dist, "log", False):
        lo, hi, value = math.log(lo), math.log(hi), math.log(value)
    return (value - lo) < 0.05 * (hi - lo) or (hi - value) < 0.05 * (hi - lo)


for name in sys.argv[2:]:
    st = optuna.load_study(study_name=name, storage=f"sqlite:///{sys.argv[1]}/{name}.db")
    tr = st.get_trials(deepcopy=False)
    states = {}
    for t in tr:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    print(f"== {name}: {len(tr)} trials {states}")
    done = [t for t in tr if t.state.name == "COMPLETE"]
    ended = [t for t in tr if t.state.name in ("COMPLETE", "PRUNED", "FAIL")]
    durs = [(t.datetime_complete - t.datetime_start).total_seconds() / 3600 for t in tr
            if t.datetime_complete and t.state.name == "COMPLETE"]
    if durs:
        print(f"   duration h (complete): median {statistics.median(durs):.2f}, max {max(durs):.2f}")
    if ended:
        bad = states.get("PRUNED", 0) + states.get("FAIL", 0)
        print(f"   pruned+failed {bad}/{len(ended)} = {100 * bad / len(ended):.0f} %")
    for t in tr:
        if t.state.name == "FAIL":
            print(f"!  FAIL #{t.number}: diverged at epoch {t.user_attrs.get('diverged_at_epoch', '?')}, {t.params}")
    best = sorted(done, key=lambda t: t.value)[:5]
    for t in best:
        extra = ""
        if "emb_eff_rank" in t.user_attrs:
            extra = f" eff_rank {t.user_attrs['emb_eff_rank']:.1f}"
        print(f"   #{t.number} value {t.value:.6g}{extra} {t.params}")
    if len(best) >= 2:
        spread = (best[-1].value - best[0].value) / abs(best[0].value)
        print(f"{'!' if spread < 0.02 else ' '}  spread best {len(best)}: {100 * spread:.1f} %"
              + (" (flat optimum, rank by seed replicates)" if spread < 0.02 else ""))
    edges = {}
    for t in best:
        for p, v in t.params.items():
            if p in t.distributions and at_edge(p, v, t.distributions[p]):
                edges[p] = edges.get(p, 0) + 1
    for p, n in edges.items():
        if n >= max(2, len(best) // 2):
            print(f"!  {p} at the edge of the search space in {n} of the best {len(best)} -> extend the range")
    for t in tr:
        iv = t.intermediate_values
        if len(iv) >= 3:
            vals = [iv[k] for k in sorted(iv)]
            if vals[-1] > 1.2 * min(vals):
                print(f"!  #{t.number} ({t.state.name}) val score rising: min {min(vals):.4g}, last {vals[-1]:.4g}")
        em = t.user_attrs.get("epoch_max") or []
        for key in sorted({k for e in em for k in e}):
            series = [e[key] for e in em if key in e and isinstance(e[key], (int, float))]
            if len(series) >= 3 and series[-1] > 10 * statistics.median(series[:-1]):
                print(f"!  #{t.number} {key} jumped: median {statistics.median(series[:-1]):.3g}, last {series[-1]:.3g}")
        if "emb_eff_rank" in t.user_attrs and "embed_dim" in t.params:
            r = t.user_attrs["emb_eff_rank"] / t.params["embed_dim"]
            if r < 0.1:
                print(f"!  #{t.number} effective rank {t.user_attrs['emb_eff_rank']:.1f} of {t.params['embed_dim']} (partial collapse)")
    running = [t for t in tr if t.state.name == "RUNNING"]
    for t in running:
        iv = t.intermediate_values
        last = max(iv) if iv else None
        print(f"   running #{t.number} since {t.datetime_start:%m-%d %H:%M} "
              f"({(now - t.datetime_start).total_seconds() / 3600:.1f} h), "
              f"epochs reported {len(iv)}, last {iv[last] if last is not None else '-'}")
    per_epoch = []
    for t in tr:
        if t.datetime_complete and t.intermediate_values:
            per_epoch.append((t.datetime_complete - t.datetime_start).total_seconds() / 3600
                             / len(t.intermediate_values))
        elif t.state.name == "RUNNING" and t.intermediate_values:
            per_epoch.append((now - t.datetime_start).total_seconds() / 3600 / len(t.intermediate_values))
    if per_epoch:
        print(f"   hours per epoch (median over trials): {statistics.median(per_epoch):.2f}")
