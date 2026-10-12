"""Reviewer point #2: does a within-account (stratified) block bootstrap
give a materially different interval from the pooled bootstrap that lets
blocks straddle the u2/u3 boundary?

The pooled bootstrap in chrono_split.py concatenates the 13 u2 differences
and 15 u3 differences into one 28-element series and draws blocks from it,
so a small number of blocks mix one account's tail with the other account's
head. This script instead draws blocks separately within each account's own
series (never crossing the boundary), concatenates the two account-specific
resamples, and reports the resulting percentile CI for comparison, at every
block length used in the paper (1-4).
"""
from pathlib import Path

import numpy as np
import chrono_split as cs

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'


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
    return out, res['n_origins']


def stratified_block_bootstrap_ci(series_list, block, reps=10000, seed=0):
    """series_list: list of 1-D arrays, one per account (stratum).
    Resample blocks independently within each stratum, then pool all
    resampled points with equal weight, exactly mirroring how the paper
    pools per-origin differences with equal weight across accounts."""
    rng = np.random.default_rng(seed)
    n_total = sum(len(s) for s in series_list)
    means = np.zeros(reps)
    for series in series_list:
        x = np.asarray(series, dtype=float)
        n = len(x)
        b = max(1, min(block, n))
        n_blocks = int(np.ceil(n / b))
        starts_max = n - b + 1
        idx = rng.integers(0, starts_max, size=(reps, n_blocks))
        offs = np.arange(b)
        samples = (idx[:, :, None] + offs[None, None, :]).reshape(reps, -1)[:, :n]
        # accumulate the (unnormalised) sum from this stratum; divide once at the end
        means += x[samples].sum(axis=1)
    means /= n_total
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    per_account = {}
    for u in USERS:
        d, n = per_origin_diffs(u)
        per_account[u] = d
        print(f'{u}: n_origins={n}')

    print(f'\n{"metric":10s} {"block":>5s} {"pooled(current)":>22s} '
          f'{"stratified(within-acct)":>26s}')
    for m in cs.METRICS:
        pooled_series = np.concatenate([per_account[u][m] for u in USERS])
        strat_series = [per_account[u][m] for u in USERS]
        for block in (1, 2, 3, 4):
            lo_p, hi_p = cs.block_bootstrap_ci(pooled_series, block, seed=0)
            lo_s, hi_s = stratified_block_bootstrap_ci(strat_series, block, seed=0)
            mean = pooled_series.mean()
            print(f'{m:10s} {block:5d}  [{lo_p:+.4f},{hi_p:+.4f}]'
                  f'          [{lo_s:+.4f},{hi_s:+.4f}]'
                  f'   (mean={mean:+.4f})')


if __name__ == '__main__':
    main()
