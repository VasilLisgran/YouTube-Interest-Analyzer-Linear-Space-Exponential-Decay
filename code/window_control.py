"""window_control.py -- is exponential decay just a smoothed hard window?

Evaluates lambda = 1.0 at several truncation ages W against the same
lambda = 1.0, W = infinity control used in chrono_split.py, on exactly the same
origins, and reports the decay condition alongside for reference.
"""
import argparse, math, sys
from collections import Counter
import numpy as np
from chrono_split import (load, place_origins, build_profile, rank_categories,
                          score_ranking, block_bootstrap_ci, sign_test_p,
                          METRICS, REGISTERED_CATEGORIES)

ap = argparse.ArgumentParser()
ap.add_argument('--users', nargs='+', required=True)
ap.add_argument('--windows', default='30,60,90,180,360')
ap.add_argument('--focus-lambda', type=float, default=0.95)
ap.add_argument('--block', type=int, default=2)
a = ap.parse_args()

Ws = [int(x) for x in a.windows.split(',')]
conds = [(f'decay lambda={a.focus_lambda}, W=inf', a.focus_lambda, None),
         ('control lambda=1.0, W=inf', 1.0, None)] + \
        [(f'window lambda=1.0, W={w}', 1.0, w) for w in Ws]
per = {c[0]: {m: [] for m in METRICS} for c in conds}
n_or = 0
print(f'candidate set: all {len(REGISTERED_CATEGORIES)} registered categories')
for path in a.users:
    ev = load(path, 'continuous_log')
    if len({e['date'] for e in ev}) < 20:
        print(f'[skip] {path}'); continue
    cuts = place_origins(ev, 'disjoint', 5, 60, 30, 10)
    if len(cuts) < 3:
        print(f'[skip] {path}: {len(cuts)} origins'); continue
    n_or += len(cuts)
    print(f'{path}: {len(cuts)} origins')
    for T, tr, te in cuts:
        tg = [e['category'] for e in te]; cnt = Counter(tg)
        ideal = sorted(cnt.values(), reverse=True)[:5]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        for lab, lam, W in conds:
            r = rank_categories(build_profile(tr, T, lam, W),
                                list(REGISTERED_CATEGORIES), str(T))
            for m, v in score_ranking(r, tg, cnt, idcg).items():
                per[lab][m].append(v)

print(f'\n=== means over {n_or} origins ===')
print(f"{'condition':32s} " + ' '.join(f'{m:>9s}' for m in METRICS))
for lab, _, _ in conds:
    print(f'{lab:32s} ' + ' '.join(f'{np.mean(per[lab][m]):9.3f}' for m in METRICS))

base = {m: np.array(per['control lambda=1.0, W=inf'][m]) for m in METRICS}
print(f'\n=== per-origin difference vs the W=inf no-forgetting control '
      f'(block bootstrap, block={a.block}) ===')
for lab, _, _ in conds:
    if lab.startswith('control'): continue
    print(f'-- {lab}')
    for m in METRICS:
        d = np.array(per[lab][m]) - base[m]
        lo, hi = block_bootstrap_ci(d, a.block)
        p, k, n = sign_test_p(list(d))
        flag = '  <- excludes zero' if (lo > 0 or hi < 0) else ''
        print(f'   {m:9s} mean={d.mean():+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  '
              f'better={k}/{n}  p_sign={p:.4f}{flag}')
