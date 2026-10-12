"""multiplicity.py -- multiplicity adjustment for the four-metric confirmatory comparison.

Reads the per-origin results written by chrono_split.py (--out), pools the 28
origins of the qualifying accounts, and reports for lambda = 0.95 vs the
no-decay control (lambda = 1.0, W = infinity):
  * the ordinary 95% and the Bonferroni-adjusted 98.75% (= 1 - 0.05/4)
    moving-block bootstrap intervals (block = 2, B = 10000, seed 0);
  * the exact sign-test p-values and their Holm-adjusted values.
"""
import json, math, sys
import numpy as np
import chrono_split as cs

path = sys.argv[1] if len(sys.argv) > 1 else 'chrono_split.json'
res = json.load(open(path))['per_account']
LAM, BLOCK, M = 0.95, 2, len(cs.METRICS)


def ci(diffs, level, reps=10000, seed=0):
    x = np.asarray(diffs, float); n = len(x); b = BLOCK
    nb = int(math.ceil(n / b)); smax = n - b + 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, smax, size=(reps, nb))
    s = (idx[:, :, None] + np.arange(b)[None, None, :]).reshape(reps, -1)[:, :n]
    m = x[s].mean(axis=1); a = (1 - level) / 2 * 100
    return float(np.percentile(m, a)), float(np.percentile(m, 100 - a))


rows = []
for m in cs.METRICS:
    d = []
    for r in res.values():
        d += list(np.array(r['per_origin'][str(LAM)][m]) - np.array(r['per_origin']['1.0'][m]))
    p, k, n = cs.sign_test_p(d)
    rows.append((m, len(d), float(np.mean(d)), ci(d, 0.95), ci(d, 1 - 0.05 / M), p, k, n))

order = sorted(range(M), key=lambda i: rows[i][5])
holm = [None] * M; run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (M - rank) * rows[i][5])); holm[i] = run

print(f'lambda={LAM} vs control, W=inf, pooled origins, block={BLOCK}, B=10000, seed=0, {M} metrics')
print(f'{"metric":9s} {"n":>3s} {"mean":>8s} {"95% CI":>19s} {"98.75% CI (Bonferroni)":>25s} {"p_sign":>8s} {"p_Holm":>8s}')
for (m, n, mu, c95, cb, p, k, nn), h in zip(rows, holm):
    print(f'{m:9s} {n:3d} {mu:+8.3f} [{c95[0]:+.3f},{c95[1]:+.3f}] [{cb[0]:+.3f},{cb[1]:+.3f}]'.ljust(70) + f' {p:8.4f} {h:8.4f}')
