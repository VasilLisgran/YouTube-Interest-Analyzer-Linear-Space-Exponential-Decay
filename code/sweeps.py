import argparse
import json
import re
from collections import defaultdict
from datetime import date

import warnings

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from pipeline_common import WEIGHT_SCHEMES, load_events

warnings.filterwarnings('ignore')

_MODEL = {}

def load(path, weight_scheme='continuous_log'):
    return load_events(path, weight_scheme=weight_scheme, dedup_key='video_id')

def embed(texts, kind='labse'):
    if kind == 'labse':
        if 'm' not in _MODEL:
            from sentence_transformers import SentenceTransformer
            print('  [loading LaBSE once]')
            _MODEL['m'] = SentenceTransformer('sentence-transformers/LaBSE',
                                              device='cpu')
        X = _MODEL['m'].encode(texts, show_progress_bar=False)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                            min_df=2).fit_transform(texts).toarray()
    return normalize(np.asarray(X, dtype=float))

def score_labels(X, labels):
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    noise = float((~mask).mean())
    sil = None
    if mask.sum() >= 2 and n_clusters >= 2:
        sil = float(silhouette_score(X[mask], labels[mask], metric='cosine'))
    return n_clusters, noise, sil

def mode_clustering(args):
    eps_grid = [float(e) for e in args.eps_grid.split(',')]
    print(f'eps grid: {eps_grid}   min_samples={args.min_samples}')
    print(f'{"account":10s} {"method":16s} {"clusters":>9s} {"noise":>7s} '
          f'{"silhouette":>11s} {"coverage":>9s}')
    print('-' * 70)

    summary = {}
    for path in args.users:
        acct = path.split('/')[-1].replace('user_videos_', '').replace('.json', '')
        videos = load(path, args.weight_scheme)
        if len(videos) < args.min_videos:
            print(f'{acct:10s} [skip] only {len(videos)} usable videos')
            continue

        by_cat = defaultdict(list)
        for v in videos:
            by_cat[v['category']].append(v)

        rows = {}
        # embed each category once, reuse for every eps
        cat_emb = {c: embed([v['clean'] for v in items], args.embedder)
                   for c, items in by_cat.items() if len(items) >= 2}

        for eps in eps_grid:
            sils, noises, nclust, npts = [], [], 0, 0
            for c, X in cat_emb.items():
                labels = DBSCAN(eps=eps, min_samples=args.min_samples,
                                metric='cosine').fit_predict(X)
                k, noise, sil = score_labels(X, labels)
                nclust += k
                npts += len(X)
                noises.append(noise)
                if sil is not None:
                    sils.append(sil)
            rows[f'DBSCAN eps={eps}'] = (
                nclust, float(np.mean(noises)),
                float(np.mean(sils)) if sils else None,
                1 - float(np.mean(noises)))

        # HDBSCAN: density chosen locally, no global eps
        sils, noises, nclust = [], [], 0
        for c, X in cat_emb.items():
            if len(X) < args.min_samples + 1:
                noises.append(1.0)
                continue
            labels = HDBSCAN(min_cluster_size=max(2, args.min_samples),
                             metric='euclidean', copy=True).fit_predict(X)
            k, noise, sil = score_labels(X, labels)
            nclust += k
            noises.append(noise)
            if sil is not None:
                sils.append(sil)
            if args.verbose:
                s = f'{sil:.3f}' if sil is not None else 'n/a'
                print(f'    [hdbscan] {c[:22]:22s} n={len(X):5d} '
                      f'clusters={k:4d} noise={noise:.2f} sil={s}')
        rows['HDBSCAN'] = (nclust, float(np.mean(noises)),
                           float(np.mean(sils)) if sils else None,
                           1 - float(np.mean(noises)))

        for name, (k, noise, sil, cov) in rows.items():
            s = f'{sil:.3f}' if sil is not None else '  n/a'
            print(f'{acct:10s} {name:16s} {k:9d} {noise:7.2f} {s:>11s} {cov:9.2f}')
        print()
        summary[acct] = {k: {'clusters': v[0], 'noise': v[1],
                             'silhouette': v[2], 'coverage': v[3]}
                         for k, v in rows.items()}

    json.dump(summary, open('sweep_clustering.json', 'w'), indent=2)
    print('written: sweep_clustering.json')


def mode_lambda(args):
    lam_grid = [float(x) for x in args.lambda_grid.split(',')]
    ref = date.fromisoformat(args.reference_date)
    print(f'lambda grid: {lam_grid}   reference={ref}   window={args.max_days}d'
          f'   weight_scheme={args.weight_scheme}')
    print()

    summary = {}
    for path in args.users:
        acct = path.split('/')[-1].replace('user_videos_', '').replace('.json', '')
        videos = load(path, args.weight_scheme)

        by_cat = defaultdict(list)
        skipped = 0
        for v in videos:
            if v.get('date') is None or v.get('weight') is None:
                skipped += 1
                continue
            d = (ref - v['date']).days
            if d < 0 or d > args.max_days:
                continue
            by_cat[v['category']].append((v['weight'], d))

        if not by_cat:
            print(f'{acct}: no dated events in window (skipped {skipped})')
            continue

        # Detect burst accounts: if every like shares one date, the decay term
        # is a constant factor and cancels in the normalized profile. Such an
        # account cannot support any claim about temporal decay.
        all_days = sorted({d for items in by_cat.values() for _, d in items})
        span = max(all_days) - min(all_days) if all_days else 0
        burst = span <= args.burst_days

        print(f'--- {acct} ---  distinct like-dates in window: {len(all_days)}, '
              f'span: {span}d' + ('   [BURST: no temporal signal]' if burst else ''))
        header = f'{"lambda":>8s} {"half-life":>10s} {"mean N_eff":>11s} {"mean N_eff/n":>13s}'
        print(header)
        rows = {}
        for lam in lam_grid:
            ratios, neffs = [], []
            for c, items in by_cat.items():
                w = np.array([wt * (lam ** d) for wt, d in items], dtype=float)
                tot, sq = w.sum(), (w * w).sum()
                if sq <= 0:
                    continue
                neff = tot * tot / sq
                neffs.append(neff)
                ratios.append(neff / len(items))
            hl = np.log(2) / -np.log(lam)
            rows[lam] = {'half_life_days': float(hl),
                         'mean_n_eff': float(np.mean(neffs)),
                         'mean_ratio': float(np.mean(ratios))}
            print(f'{lam:8.3f} {hl:9.1f}d {np.mean(neffs):11.2f} '
                  f'{np.mean(ratios):13.3f}')
        print()
        summary[acct] = {'burst': burst, 'span_days': span,
                         'distinct_dates': len(all_days), 'by_lambda': rows}

    json.dump(summary, open('sweep_lambda.json', 'w'), indent=2)
    burst_accts = [a for a, v in summary.items() if v['burst']]
    if burst_accts:
        print('BURST accounts (exclude from any temporal-decay claim): '
              + ', '.join(burst_accts))
    print('written: sweep_lambda.json')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['clustering', 'lambda'])
    ap.add_argument('--users', nargs='+', required=True)
    ap.add_argument('--embedder', default='labse', choices=['labse', 'char'])
    ap.add_argument('--eps-grid', default='0.30,0.40,0.45,0.51,0.60,0.70')
    ap.add_argument('--min-samples', type=int, default=2)
    ap.add_argument('--min-videos', type=int, default=30)
    ap.add_argument('--lambda-grid', default='0.80,0.90,0.95,0.98,0.99,0.995,0.999')
    ap.add_argument('--reference-date', default='2026-08-26')
    ap.add_argument('--max-days', type=int, default=360)
    ap.add_argument('--verbose', action='store_true',
                    help='print per-category HDBSCAN detail, so an aggregate '
                         'that looks suspiciously stable can be checked')
    ap.add_argument('--weight-scheme', default='continuous_log',
                    choices=sorted(WEIGHT_SCHEMES))
    ap.add_argument('--burst-days', type=int, default=7,
                    help='accounts whose likes span <= this many days are '
                         'flagged as burst-liked')
    args = ap.parse_args()

    if args.mode == 'clustering':
        mode_clustering(args)
    else:
        mode_lambda(args)


if __name__ == '__main__':
    main()
