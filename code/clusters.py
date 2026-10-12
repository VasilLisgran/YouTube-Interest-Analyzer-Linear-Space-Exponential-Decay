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
    ap.add_argument('--eps', type=float, default=0.51)
    ap.add_argument('--min-samples', type=int, default=2)
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
