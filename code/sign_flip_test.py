"""Reviewer point #3: a sign-flip (randomisation) test on the 28 paired
per-origin differences, as a distribution-free alternative to the sign
test that does not throw away the magnitude of each difference.

H0: the sign of each origin's (lambda=0.95 - control) difference is an
independent fair coin flip, i.e. decay has no consistent directional
effect. Under H0, the distribution of the mean difference is symmetric
under independently flipping the sign of each of the 28 observed
differences. We enumerate this exactly when feasible (2**28 is too large,
so we Monte Carlo with a large number of random sign patterns) and report
the two-sided p-value as the fraction of |permuted mean| >= |observed mean|.
"""
from pathlib import Path

import numpy as np
import chrono_split as cs

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

REPS = 200_000


class Args:
    origins = 'disjoint'
    splits = 5
    horizon_days = 60
    min_train = 30
    min_test = 10
    min_origins = 3
    min_distinct_dates = 20
    random_repeats = 200
    max_age_days = 0


GRID = [0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
USERS = ['user_videos_u2.json', 'user_videos_u3.json']


def per_origin_diffs(path, lam=0.95):
    a = Args()
    ev = cs.load(str(DATA_DIR / path), 'continuous_log')
    res = cs.evaluate_account(ev, GRID, a, seed=0)
    out = {}
    for m in cs.METRICS:
        x = np.array(res['per_origin'][str(lam)][m], dtype=float)
        y = np.array(res['per_origin']['1.0'][m], dtype=float)
        out[m] = x - y
    return out


def sign_flip_test(diffs, reps=REPS, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(diffs, dtype=float)
    n = len(x)
    obs = x.mean()
    signs = rng.choice([-1.0, 1.0], size=(reps, n))
    perm_means = (signs * x[None, :]).mean(axis=1)
    p_two_sided = float(np.mean(np.abs(perm_means) >= abs(obs) - 1e-12))
    return obs, p_two_sided, perm_means


def main():
    per_account = {}
    for u in USERS:
        per_account[u] = per_origin_diffs(u)

    for m in cs.METRICS:
        pooled = np.concatenate([per_account[u][m] for u in USERS])
        obs, p, perm_means = sign_flip_test(pooled)
        # exact sign test for comparison (as reported in the paper)
        nz = pooled[pooled != 0]
        k = int((nz > 0).sum())
        from math import comb
        n_nz = len(nz)
        tail = sum(comb(n_nz, i) for i in range(0, min(k, n_nz - k) + 1)) / 2**n_nz
        p_sign = min(1.0, 2 * tail)
        print(f'{m:10s}  n={len(pooled):2d}  mean={obs:+.4f}  '
              f'sign-flip p={p:.4f}  (sign test p={p_sign:.4f}, '
              f'k={k}/{n_nz} nonzero)')


if __name__ == '__main__':
    main()
