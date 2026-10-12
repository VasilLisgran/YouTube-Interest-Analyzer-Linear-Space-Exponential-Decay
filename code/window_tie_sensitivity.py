"""window_tie_sensitivity.py -- window control (06) under 40 tie-break salts.
Short windows leave many categories at score zero, so the hash tie-break
decides more of the ranking than at W = infinity. This re-runs the window
comparison of window_control.py under 40 salts and reports the spread of the
pooled mean difference against the lambda = 1.0, W = infinity control."""
import math, sys
from collections import Counter
import numpy as np
import chrono_split as cs
users = sys.argv[1:]
N_SALT = 40
conds = [('decay 0.95, W=inf', 0.95, None)] + [(f'window W={w}', 1.0, w) for w in (30, 60, 90, 180, 360)]
res = {c[0]: {m: [] for m in cs.METRICS} for c in conds}
zero_share = {c[0]: [] for c in conds}
for salt in range(N_SALT):
    cs.TIE_SALT = f's{salt}'
    diffs = {c[0]: {m: [] for m in cs.METRICS} for c in conds}
    for path in users:
        ev = cs.load(path, 'continuous_log')
        if len({e['date'] for e in ev}) < 20: continue
        cuts = cs.place_origins(ev, 'disjoint', 5, 60, 30, 10)
        if len(cuts) < 3: continue
        for T, tr, te in cuts:
            tg = [e['category'] for e in te]; cnt = Counter(tg)
            ideal = sorted(cnt.values(), reverse=True)[:5]
            idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
            cats = list(cs.REGISTERED_CATEGORIES)
            base = cs.score_ranking(cs.rank_categories(cs.build_profile(tr, T, 1.0, None), cats, str(T)), tg, cnt, idcg)
            for lab, lam, W in conds:
                prof = cs.build_profile(tr, T, lam, W)
                if salt == 0:
                    zero_share[lab].append(sum(1 for c in cats if prof.get(c, 0.0) == 0.0) / len(cats))
                sc = cs.score_ranking(cs.rank_categories(prof, cats, str(T)), tg, cnt, idcg)
                for m in cs.METRICS: diffs[lab][m].append(sc[m] - base[m])
    for lab in diffs:
        for m in cs.METRICS: res[lab][m].append(float(np.mean(diffs[lab][m])))
print(f'pooled mean difference vs lambda=1.0, W=inf control, 28 origins, {N_SALT} tie-break salts')
print(f"{'condition':18s} {'metric':9s} {'mean':>8s} {'sd':>7s} {'min':>8s} {'max':>8s} {'salts<0':>8s}")
for lab, _, _ in conds:
    for m in cs.METRICS:
        v = np.array(res[lab][m])
        print(f'{lab:18s} {m:9s} {v.mean():+8.4f} {v.std(ddof=1):7.4f} {v.min():+8.4f} {v.max():+8.4f} {int((v<0).sum()):>5d}/{N_SALT}')
print('\nfor reference, the default salt used in 06_window_control.txt (TIE_SALT = \'\'):')
cs.TIE_SALT = ''
ref = {c[0]: {m: [] for m in cs.METRICS} for c in conds}
for path in users:
    ev = cs.load(path, 'continuous_log')
    if len({e['date'] for e in ev}) < 20: continue
    cuts = cs.place_origins(ev, 'disjoint', 5, 60, 30, 10)
    if len(cuts) < 3: continue
    for T, tr, te in cuts:
        tg = [e['category'] for e in te]; cnt = Counter(tg)
        ideal = sorted(cnt.values(), reverse=True)[:5]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        cats = list(cs.REGISTERED_CATEGORIES)
        base = cs.score_ranking(cs.rank_categories(cs.build_profile(tr, T, 1.0, None), cats, str(T)), tg, cnt, idcg)
        for lab, lam, W in conds:
            sc = cs.score_ranking(cs.rank_categories(cs.build_profile(tr, T, lam, W), cats, str(T)), tg, cnt, idcg)
            for m in cs.METRICS: ref[lab][m].append(sc[m] - base[m])
for lab, _, _ in conds:
    print(f'  {lab:18s} ' + '  '.join(f'{m} {np.mean(ref[lab][m]):+.4f}' for m in cs.METRICS))
print('\nmean share of registered categories with zero score (salt-independent):')
for lab in zero_share:
    if zero_share[lab]: print(f'  {lab:18s} {np.mean(zero_share[lab]):.3f}')
