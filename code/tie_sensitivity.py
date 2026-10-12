"""Tie-breaking sensitivity for the confirmatory comparison.

Re-runs the whole rolling-origin experiment under independent seeds of the
tie-breaking jitter and reports the pooled lambda=0.95-vs-control difference
each time. Uses chrono_split.py unchanged.
"""
import argparse
import numpy as np
import chrono_split as cs


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--users', nargs='+', required=True)
    ap.add_argument('--seeds', type=int, default=40)
    ap.add_argument('--block', type=int, default=2)
    ap.add_argument('--focus', type=float, default=0.95)
    args = ap.parse_args()
    a = Args()

    rows = {m: [] for m in cs.METRICS}
    los = {m: [] for m in cs.METRICS}
    his = {m: [] for m in cs.METRICS}
    print(f'{"seed":>5}' + ''.join(f'{m:>26}' for m in cs.METRICS))
    for s in range(args.seeds):
        pooled = {m: [] for m in cs.METRICS}
        for p in args.users:
            ev = cs.load(p, 'continuous_log')
            if len({e['date'] for e in ev}) < a.min_distinct_dates:
                continue
            r = cs.evaluate_account(ev, GRID, a, seed=s)
            if not r or not r.get('ok'):
                continue
            for m in cs.METRICS:
                x = np.array(r['per_origin'][str(args.focus)][m], float)
                y = np.array(r['per_origin']['1.0'][m], float)
                pooled[m].extend(list(x - y))
        line = f'{s:5d}'
        for m in cs.METRICS:
            d = np.array(pooled[m])
            lo, hi = cs.block_bootstrap_ci(d, args.block, seed=0)
            rows[m].append(d.mean()); los[m].append(lo); his[m].append(hi)
            line += f'  {d.mean():+.4f} [{lo:+.3f},{hi:+.3f}]'
        print(line)

    print(f'\n--- range over {args.seeds} independent tie-breaking seeds '
          f'(n=28 origins) ---')
    for m in cs.METRICS:
        print(f'  {m:9s} mean diff in [{min(rows[m]):+.4f},{max(rows[m]):+.4f}]'
              f'   CI lower in [{min(los[m]):+.4f},{max(los[m]):+.4f}]'
              f'   CI upper in [{min(his[m]):+.4f},{max(his[m]):+.4f}]'
              f'   CI excludes 0 in {sum(1 for l in los[m] if l > 0)}'
              f'/{args.seeds} seeds')


if __name__ == '__main__':
    main()
