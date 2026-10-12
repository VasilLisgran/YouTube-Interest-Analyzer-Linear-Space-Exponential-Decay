"""Window control (Section 5.6): is exponential decay just a smoothed hard window?

Every condition is evaluated on exactly the origins of the main experiment and
compared against the same lambda=1.0, W=inf no-forgetting control. All loading,
profile building, ranking, scoring and bootstrapping is done by the functions of
chrono_split.py, unchanged, so this harness adds no preprocessing of its own.

Because truncation drives many category scores to exactly zero, the ordering of
those categories is decided by the tie-breaking jitter rather than by the data.
The window rows are therefore seed-dependent in a way the decay row is not, and
--seeds reports the range over independent seeds alongside the seed-0 run.

Usage:
    python window_control.py --users ../data/user_videos_*.json --seeds 8
"""
import argparse

import numpy as np

import chrono_split as cs


class Args:
    """The protocol of Section 5.1, fixed."""
    origins = 'disjoint'
    splits = 5
    horizon_days = 60
    min_train = 30
    min_test = 10
    min_origins = 3
    min_distinct_dates = 20
    random_repeats = 200
    max_age_days = 0


CONDITIONS = [
    ('decay,  lam=0.95, W=inf', 0.95, 0),
    ('window, lam=1.00, W=360', 1.0, 360),
    ('window, lam=1.00, W=180', 1.0, 180),
    ('window, lam=1.00, W=90', 1.0, 90),
    ('window, lam=1.00, W=60', 1.0, 60),
    ('window, lam=1.00, W=30', 1.0, 30),
]


def run(path, lam, max_age_days, seed, weight_scheme='continuous_log'):
    """Evaluate one (lambda, W) condition on one account. None if it is skipped."""
    a = Args()
    a.max_age_days = max_age_days
    events = cs.load(path, weight_scheme)
    if len({e['date'] for e in events}) < a.min_distinct_dates:
        return None
    res = cs.evaluate_account(events, [lam], a, seed=seed)
    if res is None or not res.get('ok'):
        return None
    return res


def pooled_diffs(users, lam, W, seed, base):
    """Per-origin differences against the untruncated no-decay control."""
    out = {m: [] for m in cs.METRICS}
    for p in users:
        r = run(p, lam, W, seed)
        for m in cs.METRICS:
            a = np.array(r['per_origin'][str(lam)][m], dtype=float)
            b = np.array(base[p]['per_origin']['1.0'][m], dtype=float)
            out[m].extend(list(a - b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--users', nargs='+', required=True)
    ap.add_argument('--block', type=int, default=2)
    ap.add_argument('--seeds', type=int, default=8,
                    help='number of independent tie-breaking seeds to report')
    args = ap.parse_args()

    qualifying = [p for p in args.users if run(p, 1.0, 0, 0) is not None]
    skipped = [p for p in args.users if p not in qualifying]
    for p in skipped:
        print(f'[skip] {p}: does not meet the protocol requirements')

    base0 = {p: run(p, 1.0, 0, 0) for p in qualifying}
    n_origins = [base0[p]['n_origins'] for p in qualifying]
    print(f'control: lambda=1.0, W=inf  |  accounts={len(qualifying)}  '
          f'origins={n_origins}  total={sum(n_origins)}')
    print('all conditions evaluated on exactly these origins\n')

    ranges = {}
    for seed in range(args.seeds):
        base = base0 if seed == 0 else {p: run(p, 1.0, 0, seed)
                                        for p in qualifying}
        for label, lam, W in CONDITIONS:
            d = pooled_diffs(qualifying, lam, W, seed, base)
            for m in cs.METRICS:
                x = np.array(d[m])
                lo, hi = cs.block_bootstrap_ci(x, args.block, seed=seed)
                ranges.setdefault((label, m), []).append((x.mean(), lo, hi))

    print(f'{"condition":24}' + ''.join(f'{m:>26}' for m in cs.METRICS))
    for label, lam, W in CONDITIONS:
        row = f'{label:24}'
        for m in cs.METRICS:
            mu, lo, hi = ranges[(label, m)][0]
            row += f'  {mu:+.3f} [{lo:+.3f},{hi:+.3f}]'
        print(row)
    print(f'\nseed-0 run; 95% moving-block bootstrap, block={args.block}, '
          f'B=10000')

    print(f'\n--- range of the mean difference over {args.seeds} '
          f'independent tie-breaking seeds ---')
    print(f'{"condition":24}' + ''.join(f'{m:>22}' for m in cs.METRICS))
    for label, lam, W in CONDITIONS:
        row = f'{label:24}'
        for m in cs.METRICS:
            v = [x[0] for x in ranges[(label, m)]]
            row += f'  {min(v):+.3f} .. {max(v):+.3f}'
        print(row)
    print('\nThe decay row is seed-invariant; the window rows are not, because '
          'truncation\nsends many category scores to exactly zero and their '
          'order is then decided by\nthe tie-breaking jitter. See Section 5.6.')


if __name__ == '__main__':
    main()
