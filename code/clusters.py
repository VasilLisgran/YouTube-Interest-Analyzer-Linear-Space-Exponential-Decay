"""
clusters.py — per-category semantic clustering of liked-video titles.

Changes relative to the pilot client:
  [FIX-1] Input and output paths are CLI arguments, so six accounts can be
          processed without editing the file between runs.
  [FIX-2] Keyword ranking now uses tf * idf, as described in Section 3.3 of the
          manuscript. The previous version ranked by raw frequency only, so the
          manuscript described a method the released code did not implement.
  [FIX-3] clean_text now strips URLs and @mentions BEFORE punctuation. In the
          previous version the punctuation pass ran first, which removed the
          '//' and '@' anchors and made both regexes unreachable.
  [FIX-4] The stop-word list no longer contains corpus-specific tokens
          ('woman', 'cry', 'birds', 'nooo', 'dahaka', ...). Hand-removing words
          after inspecting one's own data injects analyst knowledge of the
          labels into the model and is not defensible in a paper about
          evaluation hygiene. Only closed-class function words, platform
          boilerplate and calendar terms remain.
  [FIX-5] eps, min_samples and top_k are CLI arguments so the paper can report
          a sensitivity analysis instead of three unexplained constants.
  [FIX-6] A per-run summary (cluster counts, noise fraction, silhouette) is
          emitted; silhouette is an intrinsic measure of cluster quality and is
          the appropriate primary metric for a clustering component, rather
          than the downstream classification proxy.
  [FIX-7] HDBSCAN is the default algorithm. The eps sweep showed that a single
          global eps cannot serve categories differing by two orders of
          magnitude in size: at eps = 0.51 cluster coverage ranged from 0.54 on
          the largest account down to 0.08 on the smallest, and coverage is what
          caps the downstream keyword method's fallback rate. HDBSCAN selects
          density locally. DBSCAN remains available as --algorithm dbscan.
  - Text cleaning, stop words, deduplication (by video_id) and keyword
          scoring now come from pipeline_common, shared with evaluate_v2.py,
          chrono_split.py, sweeps.py and decay_diagnostics.py, so the scripts
          cannot drift apart again.

Usage:
    python3 clusters.py --input ../data/user_videos_u1.json \
                        --output ../data/clusters_result_u1.json
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

# Text cleaning, stop words, deduplication and keyword scoring are
# defined once, in pipeline_common, and imported by every script in the
# pipeline.  Previously each script carried its own copy and they drifted.
from pipeline_common import (STOP_WORDS, category_document_frequency,
                             keyword_score, load_events)

MODEL_NAME = 'sentence-transformers/LaBSE'


def load(path):
    return load_events(path, dedup_key='video_id')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--eps', type=float, default=0.51)          # [FIX-5]
    ap.add_argument('--min-samples', type=int, default=2)
    # [FIX-7] Algorithm choice; see the module docstring.
    ap.add_argument('--algorithm', default='hdbscan',
                    choices=['hdbscan', 'dbscan'])
    ap.add_argument('--top-k', type=int, default=5)
    ap.add_argument('--model', default=MODEL_NAME)
    args = ap.parse_args()

    videos = load(args.input)
    by_category = defaultdict(list)
    for v in videos:
        by_category[v['category']].append(v)

    df = category_document_frequency(by_category)
    n_cat = len(by_category)

    model = SentenceTransformer(args.model)
    result, report = {}, []

    for category, items in sorted(by_category.items()):
        if len(items) < 2:
            report.append((category, len(items), 0, 0.0, None))
            continue

        emb = model.encode([v['clean'] for v in items], show_progress_bar=False)
        if args.algorithm == 'hdbscan':
            if len(items) < max(2, args.min_samples) + 1:
                labels = np.full(len(items), -1)
            else:
                labels = HDBSCAN(min_cluster_size=max(2, args.min_samples),
                                 metric='euclidean',
                                 copy=True).fit_predict(np.asarray(emb))
        else:
            labels = DBSCAN(eps=args.eps, min_samples=args.min_samples,
                            metric='cosine').fit_predict(emb)

        # [FIX-6] intrinsic quality, computed on non-noise points only
        sil = None
        mask = labels != -1
        if mask.sum() >= 2 and len(set(labels[mask])) >= 2:
            sil = float(silhouette_score(np.asarray(emb)[mask], labels[mask],
                                         metric='cosine'))

        grouped = defaultdict(list)
        for v, lab in zip(items, labels):
            if lab != -1:
                grouped[int(lab)].append(v)

        keywords = {}
        for lab, group in sorted(grouped.items()):
            tf = Counter()
            for v in group:
                for w in v['clean'].split():
                    if w not in STOP_WORDS and len(w) > 2:
                        tf[w] += 1
            # [FIX-2] tf * idf over categories, deterministic tie-break by word
            top = keyword_score(tf, df, n_cat, args.top_k)
            if top:
                keywords[str(lab)] = top

        if keywords:
            result[category] = keywords

        noise = float((labels == -1).mean())
        report.append((category, len(items), len(grouped), noise, sil))

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f'{len(videos)} videos, {n_cat} categories, '
          f'algorithm={args.algorithm} -> {args.output}')
    print(f'{"category":22s} {"n":>4s} {"clusters":>9s} {"noise":>7s} {"silhouette":>11s}')
    for cat, n, k, noise, sil in report:
        s = f'{sil:.3f}' if sil is not None else '  n/a'
        print(f'{cat:22s} {n:4d} {k:9d} {noise:7.2f} {s:>11s}')


if __name__ == '__main__':
    main()
