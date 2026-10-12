"""Effective sample size in the configuration Experiment 1 actually runs in.

sweeps.py --mode lambda sweeps N_eff at the client's default truncation of
W = 360 days and over a grid that stops at lambda = 0.999. Section 5 of the
paper runs at W = inf and against an explicit lambda = 1.0 control, so the
figures quoted when interpreting Section 5 should come from that
configuration rather than from the truncated sweep.

This script recomputes Kish's N_eff with the same per-category formula used
in sweeps.py, but at W = inf and with lambda = 1.0 included, loading events
through chrono_split.py so the weights are the ones the experiment used.

Usage:
    python neff_untruncated.py --users ../data/user_videos_u2.json ...
"""
import argparse
import math
from collections import defaultdict

import numpy as np

import chrono_split as cs

GRID = [0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--users', nargs='+', required=True)
    ap.add_argument('--weight-scheme', default='continuous_log')
    ap.add_argument('--min-distinct-dates', type=int, default=20)
    args = ap.parse_args()

    print(f'Kish N_eff, W=inf (no truncation), weight_scheme='
          f'{args.weight_scheme}')
    print('aggregation: unweighted arithmetic mean over the categories of an '
          'account\n')

    for path in args.users:
        events = cs.load(path, args.weight_scheme)
        dates = {e['date'] for e in events}
        if len(dates) < args.min_distinct_dates:
            print(f'[skip] {path}: {len(dates)} distinct like-dates')
            continue
        ref = max(e['date'] for e in events)

        by_cat = defaultdict(list)
        for e in events:
            age = (ref - e['date']).days
            if age < 0:
                continue
            by_cat[e['category']].append((e['weight'], age))

        print(f'--- {path} ---  reference={ref}  categories={len(by_cat)}  '
              f'events={len(events)}')
        print(f'{"lambda":>8s} {"half-life":>11s} {"mean N_eff":>11s}'
              f' {"mean N_eff/n":>13s}')
        for lam in GRID:
            neffs, ratios = [], []
            for items in by_cat.values():
                w = np.array([wt * (lam ** d) for wt, d in items], dtype=float)
                tot, sq = w.sum(), (w * w).sum()
                if sq <= 0:
                    continue
                neff = tot * tot / sq
                neffs.append(neff)
                ratios.append(neff / len(items))
            hl = 'inf' if lam == 1.0 else f'{math.log(2) / -math.log(lam):.1f}d'
            tag = '  <- no-decay control' if lam == 1.0 else ''
            print(f'{lam:8.3f} {hl:>11s} {np.mean(neffs):11.2f}'
                  f' {np.mean(ratios):13.3f}{tag}')
        print()


if __name__ == '__main__':
    main()
