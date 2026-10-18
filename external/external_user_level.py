"""external_user_level.py -- user-level analysis of chrono_split.json for the external replication.

Reads the per-user results written by chrono_split.py, takes each user's mean per-origin difference
(lambda 0.95 minus the lambda 1.0 control) and summarises over users, as in
PROTOCOL_external_replication.md section 5.
  python3 external_user_level.py chrono_split_external.json [--drop-ids 213 ...]

--drop-ids removes the listed user identifiers before the analysis (used for the duplicated
history of users 118 and 213).
"""
import json, math, sys
import numpy as np
import os
for d in ('../code', '../PythonSystem'):   # works in this package and in the main repo layout
    if os.path.exists(os.path.join(d, 'chrono_split.py')):
        sys.path.insert(0, d); break
import chrono_split as cs

LAM = 0.95
args = sys.argv[1:]
drop = set()
if '--drop-ids' in args:
    i = args.index('--drop-ids'); drop = set(args[i + 1:]); args = args[:i]
path = args[0]
res = json.load(open(path))['per_account']
users = sorted(u for u in res if u not in drop)
if drop:
    print(f'dropped identifiers: {sorted(drop)}')
print(f'users with results: {len(users)}')
if not users:
    sys.exit(0)

rows, adj = [], {}
for m in cs.METRICS:
    d = np.array([np.mean(np.array(res[u]['per_origin'][str(LAM)][m]) -
                          np.array(res[u]['per_origin']['1.0'][m])) for u in users])
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p, k, n = cs.sign_test_p(list(d))
    try:
        from scipy.stats import wilcoxon
        pw = float(wilcoxon(d[d != 0]).pvalue) if (d != 0).sum() > 0 else 1.0
    except Exception:
        pw = float('nan')
    rows.append((m, float(d.mean()), float(lo), float(hi), float(d.std(ddof=1)) if len(d) > 1 else float('nan'),
                 k, n, p, pw))

M = len(rows); order = sorted(range(M), key=lambda i: rows[i][7]); holm = [0] * M; run = 0.0
for r, i in enumerate(order):
    run = max(run, min(1.0, (M - r) * rows[i][7])); holm[i] = run
print(f'{"metric":8s} {"mean":>8s} {"95% CI (users)":>20s} {"SD":>7s} {"pos/nonzero":>12s} '
      f'{"p_sign":>8s} {"Holm":>7s} {"p_Wilcox":>9s}')
for i, (m, mu, lo, hi, sd, k, n, p, pw) in enumerate(rows):
    print(f'{m:8s} {mu:+8.4f} [{lo:+.4f},{hi:+.4f}] {sd:7.4f} {k:5d}/{n:<6d} {p:8.4f} {holm[i]:7.4f} {pw:9.4f}')


# ---------------------------------------------------------------------------
# Secondary analyses (protocol section 5): whole lambda grid at the user level,
# and the primary metric by tertile of history length (number of origins).
# ---------------------------------------------------------------------------
def user_diffs(lam, m, subset=None):
    us = users if subset is None else subset
    return np.array([np.mean(np.array(res[u]['per_origin'][str(lam)][m]) -
                             np.array(res[u]['per_origin']['1.0'][m])) for u in us])


def boot_ci(d, seed=0):
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    return np.percentile(b, [2.5, 97.5])


grid = sorted(float(k) for k in res[users[0]]['per_origin'] if k not in ('random', '1.0'))
print('\n--- user-level mean difference vs control, whole grid (secondary) ---')
print(f'{"lambda":>7s} {"half-life":>9s} ' + ' '.join(f'{m:>24s}' for m in cs.METRICS))
for lam in grid:
    hl = math.log(2) / -math.log(lam)
    cells = []
    for m in cs.METRICS:
        d = user_diffs(lam, m); lo, hi = boot_ci(d)
        cells.append(f'{d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]')
    print(f'{lam:7.3f} {hl:8.1f}d ' + ' '.join(f'{c:>24s}' for c in cells))

n_or = np.array([res[u]['n_origins'] for u in users])
t1, t2 = np.percentile(n_or, [100 / 3, 200 / 3])
print(f'\n--- recall@3 at lambda={LAM} by tertile of number of origins (cuts {t1:.1f}, {t2:.1f}) ---')
for name, mask in (('short', n_or <= t1), ('middle', (n_or > t1) & (n_or <= t2)), ('long', n_or > t2)):
    if not mask.any():
        print(f'  {name:7s} n=  0'); continue
    sub = [u for u, k in zip(users, mask) if k]
    d = user_diffs(LAM, 'recall@3', sub); lo, hi = boot_ci(d)
    print(f'  {name:7s} n={len(sub):3d}  origins {n_or[mask].min()}-{n_or[mask].max()}  '
          f'mean {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  positive {int((d > 0).sum())}/{int((d != 0).sum())}')
